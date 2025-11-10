# Pest Detection — Interactive README

This repository contains a complete pest-detection system: a React frontend, a FastAPI + PyTorch backend, and Arduino integration for irrigation + pest buzzer control.

Quick links
- Frontend: `web-frontend/`
- Backend: `web-backend/`
- Arduino examples: `arduino/` (includes `smart_irrigation_pest_auto_off.ino`)

Table of contents
- Features
- Prerequisites
- Quick start (backend & frontend)
- Arduino wiring & sketch
- API: /predict and /pest (examples)
- Troubleshooting & FAQ
- Contributing

## Features
- Upload pest images from the UI and get model predictions.
- Automatic buzzer control via serial when pests are detected.
- Irrigation motor control (relay) driven by moisture sensor.
- Debug-friendly: backend logs Arduino serial messages, and Arduino prints readable debug lines.

## Prerequisites
- Node.js (for frontend)
- Python 3.11 (recommended) and a virtualenv for the backend
- Arduino IDE (to upload sketches) and a USB cable
- A piezo buzzer (active or passive) and a relay module

## Quick start — Backend (recommended using project venv)

1. Create and activate the virtualenv (Windows PowerShell example):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install Python dependencies into the venv:

```powershell
python -m pip install -r web-backend/requirements.txt
# If requirements.txt is missing, install at least:
python -m pip install fastapi uvicorn[standard] pyserial torch torchvision pillow numpy
```

3. Set Arduino COM port (optional) and start the backend using the venv python:

```powershell
#$env:ARDUINO_PORT = 'COM9'    # set your Arduino COM port
#$env:ARDUINO_BAUD = '9600'
C:/Users/vinay/OneDrive/Desktop/Pests_CNN/.venv/Scripts/python.exe -m uvicorn web-backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Notes
- Backend prints startup logs. If it opens serial successfully you should see: `[DEBUG] Opened serial on COM9 @ 9600`.
- The backend will also spawn a reader thread and print any lines the Arduino sends (prefixed with `[ARDUINO]`).

## Quick start — Frontend

1. Open a new terminal, install deps and start:

```powershell
cd web-frontend
npm install
npm start
```

2. The UI will be served at `http://localhost:3000` (or the address shown by the dev server).

## Arduino wiring & sketch

Files
- `arduino/smart_irrigation_pest_auto_off.ino` — sketch that supports `PEST_ON <seconds>` and auto-off

Wiring (piezo buzzer)
- If buzzer current is low (<20–30 mA):
  - Buzzer + -> Arduino digital pin 8 (or transistor collector)
  - Buzzer - -> GND (or transistor emitter)
- Safer (recommended for louder/12V buzzers): use an NPN transistor (2N2222)
  - Arduino pin -> 1 kΩ -> transistor base; emitter -> GND; collector -> buzzer-; buzzer+ -> +5V.

I2C LCD
- Typical addresses: `0x27` or `0x3F`. If LCD prints nothing, try switching the constructor address in the sketch.

Upload
- Open `smart_irrigation_pest_auto_off.ino` in the Arduino IDE, select board and COM port and upload.

## API: Examples

1) Predict endpoint (upload image)

curl example:

```bash
curl -X POST "http://localhost:8000/predict" -F "file=@/path/to/pest.jpg"
```

Response (example):

```json
{ "class": "armyworm", "confidence": 0.95, "pest_alert": true, "pest_alert_duration": 10 }
```

When `pest_alert` is true the backend will (by default) send the serial command `PEST_ON 10` to the Arduino and the Arduino will sound the buzzer for 10 seconds.

2) Manual pest control endpoint

POST /pest — body: `{ "pest": true, "duration": 10 }` will turn buzzer on for 10 seconds.

Example (PowerShell):

```powershell
Invoke-WebRequest -Uri http://localhost:8000/pest -Method POST -ContentType 'application/json' -Body '{"pest":true,"duration":10}' -UseBasicParsing
```

Or using the Python helper (quick test):

```python
import serial, time
ser = serial.Serial('COM9', 9600, timeout=1)
time.sleep(2)
ser.write(b'PEST_ON 8\n')  # buzzer on for 8 seconds
time.sleep(10)
ser.write(b'PEST_OFF\n')
ser.close()
```

## Troubleshooting

- Backend prints `[WARN] Failed to open serial port` → serial port busy or incorrect COM. Close Arduino Serial Monitor and restart backend.
- Backend prints `Serial not available; would send: PEST_ON` → backend couldn't open port; ensure pyserial is installed and port correct.
- Arduino prints `Unknown command` → verify the exact text sent ends with newline `\n` and format is `PEST_ON 10` or `PEST_OFF`.
- LCD blank → try address `0x3F` or check SDA/SCL pin connections.
- Buzzer silent but Arduino shows `PEST ON` → check buzzer wiring and whether the buzzer is active/passive. Try toggling `digitalWrite(buzzer_pin, HIGH)` in the sketch for active buzzers.

## FAQ

- Q: I want the backend to decide duration per prediction.
  - A: The backend sends `PEST_ON <seconds>` when auto-triggering; set `PEST_DEFAULT_DURATION` env var to change default.

- Q: How to change confidence threshold?
  - A: Set environment variable `PEST_CONFIDENCE_THRESHOLD` (default 0.8) before starting backend.

## Contributing

Pull requests welcome. Please run tests (if added) and document changes.

## License

This project is provided under the MIT License — see `LICENSE` (or add one if missing).

---

If you want, I can make the README even more interactive by adding GIFs of the UI, a short video link, or a GitHub Actions workflow badge showing the backend tests. Tell me which extras you'd like and I will add them.

## Adding new images (how to extend dataset)

1. Create a folder `new_images/` at the repository root and place your new pictures inside subfolders named by class, for example:

```
new_images/
  armyworm/
    img001.jpg
    img002.jpg
  aphids/
    img100.jpg
```

2. Run the ingestion script (from project root). This will copy files into `pest/train/<class>/` and `pest/test/<class>/` using an 80/20 split by default:

```powershell
python scripts/ingest_new_data.py --source new_images --dest pest --train-ratio 0.8
```

Options:
- `--move` : move files instead of copying
- `--train-ratio 0.9` : change split ratio
- `--seed 123` : reproducible shuffling

3. Verify the files appear under `pest/train/<class>` and `pest/test/<class>`.

4. Retrain the model using the same training script:

```powershell
# using the project venv recommended
C:/Users/vinay/OneDrive/Desktop/Pests_CNN/.venv/Scripts/python.exe pest_model/train.py --data-dir pest --epochs 30 --batch-size 16
```

Notes:
- The ingestion script will avoid filename collisions by appending a short uuid if a file with the same name already exists.
- The trainer (`pest_model/train.py`) expects the dataset root to contain `train/` and `test/` subfolders with per-class directories.


# Pest Detection Web App

This project provides an end-to-end pest detection system with a modern web interface and a FastAPI backend serving a trained CNN model.

## Features
- Upload pest images and get predictions
- Beautiful, interactive React frontend (Material UI)
- FastAPI backend serving PyTorch model
- Easy local setup

## Setup Instructions

### 1. Backend (API)
- Open a terminal and run:
  ```powershell
  cd web-backend
  C:/Users/vinay/AppData/Local/Microsoft/WindowsApps/python3.11.exe -m uvicorn main:app --reload
  ```
- Ensure `pest_model` folder and `checkpoints/best.pth` are present in the project root.

### 2. Frontend (UI)
- Open a new terminal and run:
  ```powershell
  cd web-frontend
  npm install
  npm start
  ```
- The app will open at `http://localhost:3000`.

### 3. Usage
- Upload a pest image using the web interface.
- View prediction results instantly.

## Troubleshooting
- If you see import errors, ensure you run backend from `web-backend` and frontend from `web-frontend`.
- If `npm start` fails, delete `node_modules` and run `npm install` again.
- For Python errors, ensure all dependencies are installed:
  ```powershell
  pip install fastapi uvicorn pillow numpy torch
  ```

## Customization
- To use a different model, update `web-backend/main.py` to load your checkpoint and class names.
- For UI changes, edit `web-frontend/src/App.js`.

---

Enjoy your interactive pest detection website!
