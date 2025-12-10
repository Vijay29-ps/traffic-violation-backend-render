# Colab-ready YOLO vehicle + optional helmet detection script
# Saves CSV + JSON + annotated video or image

import os, time, json, csv, uuid
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO
import requests

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

GENERAL_MODEL_PATH = "yolov8n.pt"
API_URL = "https://ig.gov-cloud.ai/bob-camunda/v1.0/camunda/execute/019afd3a-63d7-7fca-912e-bacdda266903?env=TEST&sync=false"
API_HEADERS = {"Authorization": "Bearer YOUR_TOKEN_HERE"}

CSV_HEADERS = [
    "file_name","camera_id","frame_id","vehicle_id",
    "bbox_x1","bbox_y1","bbox_x2","bbox_y2",
    "label","conf","color","lane","speed_kmph","overspeed","helmet_violation",
    "anpr_plate","anpr_confidence","hotlist_match","weapon_detected",
    "violation_type","violation_image","timestamp","meta"
]

def try_load_yolo(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"Model not found: {path}")
    model = YOLO(path)
    return model

def draw_box(img, xyxy, label, conf):
    x1,y1,x2,y2 = map(int, xyxy)
    cv2.rectangle(img,(x1,y1),(x2,y2),(255,255,0),2)
    cv2.putText(img,f"{label} {conf:.2f}",(x1,y1-4),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)

def build_record(file_name, frame_id, vehicle_id, bbox, label, conf, helmet_violation, violation_image):
    ts = datetime.utcnow().isoformat() + "Z"
    x1,y1,x2,y2 = bbox
    return {
        "file_name": file_name,
        "camera_id": "camera_0",
        "frame_id": frame_id,
        "vehicle_id": vehicle_id,
        "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
        "label": label,
        "conf": conf,
        "color": "255,255,0",
        "lane": "",
        "speed_kmph": None,
        "overspeed": False,
        "helmet_violation": helmet_violation,
        "anpr_plate": "",
        "anpr_confidence": 0,
        "hotlist_match": False,
        "weapon_detected": False,
        "violation_type": "helmet" if helmet_violation else "",
        "violation_image": violation_image,
        "timestamp": ts,
        "meta": {"detector": "yolov8"}
    }

def save_violation_crop(img, bbox):
    x1,y1,x2,y2 = map(int, bbox)
    crop = img[y1:y2, x1:x2]
    fname = f"violation_{int(time.time()*1000)}.jpg"
    out = OUTPUT_DIR / fname
    cv2.imwrite(str(out), crop)
    return str(out)

def process_image(image_path, general_model, helmet_model=None):
    img = cv2.imread(image_path)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_json = OUTPUT_DIR / f"result_{timestamp}.json"
    out_csv = OUTPUT_DIR / f"results_{timestamp}.csv"
    out_img = OUTPUT_DIR / f"annotated_{timestamp}.jpg"

    gen_res = general_model(img)[0]
    results_records = []

    for box in gen_res.boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        label = general_model.names[int(box.cls[0])]
        conf = float(box.conf[0])

        draw_box(img, xyxy, label, conf)

        helmet_violation = False
        violation_img = ""
        # (Helmet detection optional — implement similar logic)

        rec = build_record(os.path.basename(image_path), 0, uuid.uuid4().hex[:12],
                           xyxy.tolist(), label, conf, helmet_violation, violation_img)
        results_records.append(rec)

    cv2.imwrite(str(out_img), img)

    with open(out_csv, "w", newline='', encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(results_records)

    with open(out_json, "w", encoding="utf-8") as jf:
        json.dump(results_records, jf, indent=2)

    print("Saved:", out_img, out_json, out_csv)


if __name__ == "__main__":
    general_model = try_load_yolo(GENERAL_MODEL_PATH)
    print("General model loaded.")

    # Example usage:
    # python app.py image.jpg
    import sys
    if len(sys.argv) < 2:
        print("Usage: python app.py <image_or_video_path>")
        exit()

    input_path = sys.argv[1]

    if input_path.lower().endswith((".jpg",".png",".jpeg")):
        process_image(input_path, general_model)
    else:
        print("Video processing not added in this minimal GitHub version.")
