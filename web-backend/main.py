from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import numpy as np
from PIL import Image
import io
import os
import threading
import time

# Optional serial support to talk to Arduino buzzer
try:
    import serial
except Exception:
    serial = None

app = FastAPI()

# Health check endpoint
@app.get("/")
def read_root():
    return {"status": "Backend is running"}
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# Real model loader
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pest_model')))
from model import build_model
from utils import load_checkpoint
from dataset import get_default_transforms

class_names = ['aphids', 'armyworm', 'beetle', 'bollworm', 'grasshopper', 'mites', 'mosquito', 'sawfly', 'stem_borer']
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    global model
    num_classes = len(class_names)
    print("[DEBUG] Building model...")
    model = build_model('resnet50', num_classes=num_classes, pretrained=False)
    print("[DEBUG] Loading checkpoint from checkpoints/best.pth...")
    # Resolve checkpoint path relative to repository root (one level above web-backend)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    ckpt_path = os.path.join(repo_root, 'checkpoints', 'best.pth')
    print(f"[DEBUG] Resolved checkpoint path: {ckpt_path}")
    ckpt = load_checkpoint(ckpt_path, device)
    print("[DEBUG] Checkpoint loaded. Loading state dict...")
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()
    print("[DEBUG] Model ready.")

# Serial (Arduino) globals
arduino_port = os.environ.get("ARDUINO_PORT", "COM9")
arduino_baud = int(os.environ.get("ARDUINO_BAUD", 9600))
arduino_serial = None
pest_timer = None
PEST_CONFIDENCE_THRESHOLD = float(os.environ.get("PEST_CONFIDENCE_THRESHOLD", 0.8))
PEST_DEFAULT_DURATION = int(os.environ.get("PEST_DEFAULT_DURATION", 10))

def init_serial():
    global arduino_serial
    if serial is None:
        print("[WARN] pyserial not installed; serial buzzer control disabled.")
        return
    try:
        arduino_serial = serial.Serial(arduino_port, arduino_baud, timeout=1)
        # give Arduino time to reset if needed
        time.sleep(2)
        print(f"[DEBUG] Opened serial on {arduino_port} @ {arduino_baud}")
        # start background reader thread so Arduino debug lines appear in backend logs
        def _reader():
            print("[DEBUG] Arduino serial reader thread started")
            try:
                while arduino_serial and arduino_serial.is_open:
                    try:
                        line = arduino_serial.readline()
                        if not line:
                            continue
                        try:
                            decoded = line.decode(errors='ignore').strip()
                        except Exception:
                            decoded = str(line)
                        if decoded:
                            print(f"[ARDUINO] {decoded}")
                    except Exception as e:
                        print(f"[ERROR] Reading from Arduino serial: {e}")
                        break
            finally:
                print("[DEBUG] Arduino serial reader thread exiting")

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
    except Exception as e:
        arduino_serial = None
        print(f"[WARN] Failed to open serial port {arduino_port}: {e}")

def send_cmd(cmd: str):
    global arduino_serial
    try:
        if arduino_serial and arduino_serial.is_open:
            arduino_serial.write((cmd + "\n").encode())
            print(f"[DEBUG] Sent to Arduino: {cmd}")
        else:
            print(f"[WARN] Serial not available; would send: {cmd}")
    except Exception as e:
        print(f"[ERROR] Failed sending serial command {cmd}: {e}")

def send_pest_on():
    send_cmd("PEST_ON")

def send_pest_off():
    send_cmd("PEST_OFF")

@app.on_event("startup")
def startup_event():
    load_model()
    init_serial()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    print("[DEBUG] /predict called")
    try:
        image_bytes = await file.read()
        print(f"[DEBUG] Received file of size: {len(image_bytes)} bytes")
        if not image_bytes:
            print("[ERROR] No image data received.")
            return JSONResponse({"error": "No image data received."}, status_code=400)
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            print("[DEBUG] Image loaded and converted to RGB")
        except Exception as e:
            print(f"[ERROR] Failed to decode image: {e}")
            return JSONResponse({"error": f"Failed to decode image: {e}"}, status_code=400)
        img = np.array(image)
        print(f"[DEBUG] Image shape: {img.shape}")
        try:
            transform = get_default_transforms('test', 224)
            img = transform(image=img)["image"]
            print(f"[DEBUG] Transformed image shape: {img.shape}")
            img = np.transpose(img, (2, 0, 1)).astype(np.float32)
            img = torch.from_numpy(img).unsqueeze(0).to(device)
            print(f"[DEBUG] Tensor shape: {img.shape}")
            def run_inference(img_tensor):
                with torch.no_grad():
                    logits = model(img_tensor)
                    print(f"[DEBUG] Logits: {logits}")
                    probs = torch.softmax(logits, dim=1)
                    print(f"[DEBUG] Probabilities: {probs}")
                    pred = torch.argmax(probs, dim=1).item()
                    confidence = probs[0, pred].item()
                return pred, confidence, probs

            pred, confidence, probs = run_inference(img)
            pred_class = class_names[pred]
            print(f"[DEBUG] Prediction: {pred_class}, Confidence: {confidence}")

            if confidence < 0.5:
                print(f"[DEBUG] Confidence below 0.5, running inference again...")
                pred2, confidence2, probs2 = run_inference(img)
                pred_class2 = class_names[pred2]
                print(f"[DEBUG] Second prediction: {pred_class2}, Confidence: {confidence2}")
                if confidence2 > confidence:
                    pred_class, confidence, probs = pred_class2, confidence2, probs2
            response = {"class": pred_class, "confidence": round(confidence, 2)}
            if confidence < 0.5:
                response["warning"] = "Low confidence (<50%). Prediction may be unreliable."
            # --- Auto-trigger buzzer if prediction is confident ---
            try:
                if confidence >= PEST_CONFIDENCE_THRESHOLD:
                    print(f"[DEBUG] Confidence >= {PEST_CONFIDENCE_THRESHOLD}, triggering pest alarm")
                    # cancel existing timer
                    try:
                        if pest_timer and pest_timer.is_alive():
                            pest_timer.cancel()
                    except Exception:
                        pass
                    send_pest_on()
                    # schedule off after default duration
                    pest_timer = threading.Timer(PEST_DEFAULT_DURATION, send_pest_off)
                    pest_timer.start()
                    response["pest_alert"] = True
                    response["pest_alert_duration"] = PEST_DEFAULT_DURATION
                else:
                    response["pest_alert"] = False
            except Exception as e:
                print(f"[ERROR] Failed to auto-trigger pest alarm: {e}")
            return JSONResponse(response)
        except Exception as e:
            print(f"[ERROR] Exception during model inference: {e}")
            return JSONResponse({"error": f"Model inference failed: {e}"}, status_code=500)
    except Exception as e:
        print(f"[ERROR] Unexpected exception in /predict: {e}")
        return JSONResponse({"error": f"Unexpected error: {e}"}, status_code=500)


class PestRequest(BaseModel):
    pest: bool
    duration: Optional[int] = 10  # seconds to keep buzzer on (when pest=True)


@app.post("/pest")
def pest_endpoint(req: PestRequest):
    """Endpoint for frontend to notify backend about pest detection.

    JSON body: { "pest": true/false, "duration": seconds }
    When pest is true, sends PEST_ON and schedules PEST_OFF after `duration` seconds.
    When pest is false, immediately sends PEST_OFF and cancels any pending timer.
    """
    global pest_timer
    try:
        if req.pest:
            # send ON immediately
            send_pest_on()
            # cancel existing timer (if any)
            try:
                if pest_timer and pest_timer.is_alive():
                    pest_timer.cancel()
            except Exception:
                pass
            # start new timer to turn off after duration
            pest_timer = threading.Timer(req.duration, send_pest_off)
            pest_timer.start()
            return JSONResponse({"status": "pest_on", "duration": req.duration})
        else:
            # cancel timer and send off
            try:
                if pest_timer and pest_timer.is_alive():
                    pest_timer.cancel()
            except Exception:
                pass
            send_pest_off()
            return JSONResponse({"status": "pest_off"})
    except Exception as e:
        print(f"[ERROR] /pest handler failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
