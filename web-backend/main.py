from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import numpy as np
from PIL import Image
import io

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
    ckpt = load_checkpoint('../checkpoints/best.pth', device)
    print("[DEBUG] Checkpoint loaded. Loading state dict...")
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()
    print("[DEBUG] Model ready.")

@app.on_event("startup")
def startup_event():
    load_model()

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
            return JSONResponse(response)
        except Exception as e:
            print(f"[ERROR] Exception during model inference: {e}")
            return JSONResponse({"error": f"Model inference failed: {e}"}, status_code=500)
    except Exception as e:
        print(f"[ERROR] Unexpected exception in /predict: {e}")
        return JSONResponse({"error": f"Unexpected error: {e}"}, status_code=500)
