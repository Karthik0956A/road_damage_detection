from pathlib import Path
import base64

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best.pt"
HTML_PATH = BASE_DIR / "index.html"

app = FastAPI(title="Road Damage Detection")
model = YOLO(str(MODEL_PATH))


@app.get("/")
def home():
    return FileResponse(HTML_PATH)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    results = model.predict(
        source=image,
        imgsz=(640, 640),
        rect=False,
        conf=0.25,
        save=False,
        verbose=False,
    )
    result = results[0]
    annotated_image = result.plot()

    success, encoded_image = cv2.imencode(".jpg", annotated_image)
    if not success:
        raise HTTPException(status_code=500, detail="Could not encode result")

    detections = []
    names = result.names

    for box in result.boxes:
        class_id = int(box.cls[0])
        detections.append({
            "class": names[class_id],
            "confidence": round(float(box.conf[0]), 3),
            "box": [round(float(value), 1) for value in box.xyxy[0].tolist()],
        })

    image_base64 = base64.b64encode(encoded_image).decode("utf-8")
    
    return {
        "image": f"data:image/jpeg;base64,{image_base64}",
        "detections": detections,
    }