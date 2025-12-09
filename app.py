# app.py
import os
import io
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware

# Optional: ultralytics model will be lazy-loaded if available
MODEL = None
MODEL_PATH = os.getenv("MODEL_PATH", "")  # e.g. "/app/models/best.pt" or "yolov8n.pt"
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
API_USER = os.getenv("API_USER", "")
API_PASS = os.getenv("API_PASS", "")

app = FastAPI(title="Traffic Violation Backend (Render-ready)")

# Allow CORS from anywhere (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBasic()

def check_auth(credentials: HTTPBasicCredentials = Depends(security)):
    """Simple env-based Basic Auth. If no API_USER/API_PASS set, auth is skipped."""
    if not API_USER and not API_PASS:
        return True  # no auth configured
    correct_user = API_USER
    correct_pass = API_PASS
    if credentials.username != correct_user or credentials.password != correct_pass:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    return True

def load_model():
    global MODEL
    if MODEL is not None:
        return MODEL
    if not MODEL_PATH:
        return None
    try:
        # Lazy import to avoid heavy import if not used
        from ultralytics import YOLO
        MODEL = YOLO(MODEL_PATH)
        return MODEL
    except Exception as e:
        # If model cannot be loaded, set MODEL to None and log the error
        print(f"[WARN] Could not load model at {MODEL_PATH}: {e}")
        MODEL = None
        return None

@app.get("/health")
async def health():
    return {"status": "ok", "time": int(time.time())}

@app.post("/predict")
async def predict(file: UploadFile = File(...), auth: bool = Depends(check_auth)):
    # Basic file size check
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(status_code=413, detail=f"File too large ({size_mb:.2f} MB). Max is {MAX_UPLOAD_MB} MB.")
    # Save to a temp path
    temp_dir = Path("/tmp/uploads")
    temp_dir.mkdir(parents=True, exist_ok=True)
    fn = temp_dir / f"{int(time.time())}_{file.filename}"
    with open(fn, "wb") as f:
        f.write(contents)

    result = {"filename": file.filename, "size_mb": round(size_mb, 3)}

    # Attempt to run model if available
    model = load_model()
    if model is None:
        result["prediction"] = None
        result["warning"] = "No model loaded. Set MODEL_PATH env var or upload a model in the container."
        return JSONResponse(status_code=200, content=result)

    try:
        # ultralytics YOLO inference (one-liner)
        # returns results object; we convert to simple dict summary
        res = model(fn.as_posix())
        # res can be a list or Results; extract bounding boxes + class + confidence
        preds = []
        for r in res:
            boxes = r.boxes
            if boxes is None:
                continue
            for b in boxes:
                xyxy = b.xyxy.tolist()[0] if hasattr(b, "xyxy") else None
                conf = float(b.conf.tolist()[0]) if hasattr(b, "conf") else None
                cls = int(b.cls.tolist()[0]) if hasattr(b, "cls") else None
                preds.append({"xyxy": xyxy, "conf": conf, "class": cls})
        result["prediction"] = preds
    except Exception as e:
        result["error"] = f"Model inference error: {e}"
    return JSONResponse(status_code=200, content=result)

# helpful root
@app.get("/")
async def root():
    return {"message": "Traffic Violation Backend - visit /docs for API explorer"}

# Optional: endpoint to load model manually (protected)
@app.post("/_admin/load_model")
async def admin_load_model(auth: bool = Depends(check_auth)):
    loaded = load_model()
    if loaded is None:
        raise HTTPException(status_code=500, detail=f"Model not loaded. MODEL_PATH={MODEL_PATH}")
    return {"status": "model_loaded", "model_path": MODEL_PATH}
