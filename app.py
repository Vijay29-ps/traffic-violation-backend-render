# Colab-ready single cell: YOLO vehicle + optional helmet detector -> CSV + JSON (NO external API)
# Helmet model will be loaded automatically from /content/best.pt if present; no upload prompts.

# Install (Colab-friendly) packages
!pip install -q --upgrade ultralytics opencv-python-headless numpy

import os, time, json, csv, uuid, math
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO
from google.colab import files

# -----------------------------
# CONFIG (tune these)
# -----------------------------
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

GENERAL_MODEL_PATH = "yolov8n.pt"   # auto-download if missing
HELMET_MODEL_AUTOPATH = Path("/content/best.pt")  # auto-used if exists

CSV_HEADERS = [
    "file_name","camera_id","frame_id","vehicle_id",
    "bbox_x1","bbox_y1","bbox_x2","bbox_y2",
    "label","conf","color","lane","speed_kmph","overspeed","helmet_violation",
    "anpr_plate","anpr_confidence","hotlist_match","weapon_detected",
    "violation_type","violation_image","timestamp","meta","crowd_count","density_level"
]

HELMET_LABELS = ["With Helmet", "Without Helmet"]
HELMET_COLORS = {0: (0,255,0), 1:(0,0,255)}

# Speed & tracking params — calibrate px_per_meter for your scene.
PX_PER_METER = 20.0            # pixels per meter (example). Adjust per camera for accurate speed
OVERSPEED_THRESHOLD_KMPH = 40.0
MAX_MATCH_DISTANCE_PX = 120    # matching tolerance for centroid->track
SPEED_SMOOTH_ALPHA = 0.4       # smoothing for speed estimation

# Lane parameters
num_lanes = 3                  # split image width into lanes

# Crowd density thresholds (counts -> level)
density_thresholds = {
    "none": 0,
    "low": 5,
    "medium": 15
}

# -----------------------------
# HELPERS
# -----------------------------
def download_yolov8n(dest="yolov8n.pt"):
    url = "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt"
    print(f"Downloading {url} -> {dest} ...")
    import urllib.request
    urllib.request.urlretrieve(url, dest)
    print("Download complete.")

def draw_box(img, xyxy, label, conf, color):
    x1,y1,x2,y2 = map(int, xyxy)
    cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
    text = f"{label} {conf:.2f}"
    (tw,th),_ = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,0.6,2)
    cv2.rectangle(img,(x1,y1-th-4),(x1+tw,y1),color,-1)
    cv2.putText(img,text,(x1,y1-2),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)

def make_vehicle_id():
    return str(uuid.uuid4())[:12]

def save_violation_crop(img, bbox, prefix="violation"):
    x1,y1,x2,y2 = map(int, bbox)
    h, w = img.shape[:2]
    x1, y1 = max(0,x1), max(0,y1)
    x2, y2 = min(w-1, x2), min(h-1, y2)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return ""
    fname = f"{prefix}_{int(time.time()*1000)}_{make_vehicle_id()}.jpg"
    outpath = OUTPUT_DIR / fname
    cv2.imwrite(str(outpath), crop)
    return str(outpath)

def mean_color_hex(img, bbox):
    x1,y1,x2,y2 = map(int, bbox)
    h, w = img.shape[:2]
    x1, y1 = max(0,x1), max(0,y1)
    x2, y2 = min(w-1, x2), min(h-1, y2)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return "#000000"
    avg = crop.mean(axis=(0,1))  # BGR
    r,g,b = int(avg[2]), int(avg[1]), int(avg[0])
    return "#{:02X}{:02X}{:02X}".format(r,g,b)

def build_record(file_name, camera_id, frame_id, vehicle_id, bbox, label, conf, color,
                 lane="", speed_kmph=None, overspeed=False, helmet_violation=False,
                 anpr_plate="", anpr_confidence=0.0, hotlist_match=False,
                 weapon_detected=False, violation_type="", violation_image="", meta=None,
                 crowd_count=0, density_level="none"):
    ts = datetime.utcnow().isoformat() + "Z"
    x1,y1,x2,y2 = [float(x) for x in bbox]
    record = {
        "file_name": file_name,
        "camera_id": camera_id,
        "frame_id": frame_id,
        "vehicle_id": vehicle_id,
        "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
        "label": label,
        "conf": float(conf),
        "color": color if isinstance(color, str) else str(color),
        "lane": lane,
        "speed_kmph": round(float(speed_kmph),3) if speed_kmph is not None else None,
        "overspeed": bool(overspeed),
        "helmet_violation": bool(helmet_violation),
        "anpr_plate": anpr_plate,
        "anpr_confidence": float(anpr_confidence),
        "hotlist_match": bool(hotlist_match),
        "weapon_detected": bool(weapon_detected),
        "violation_type": violation_type,
        "violation_image": violation_image,
        "timestamp": ts,
        "meta": meta or {},
        "crowd_count": int(crowd_count),
        "density_level": density_level
    }
    return record

def try_load_yolo(path, model_name="model"):
    p = Path(path)
    if not p.exists():
        return None, f"{model_name} file not found at: {path}."
    if p.stat().st_size == 0:
        return None, f"{model_name} file is empty (0 bytes): {path}."
    try:
        model = YOLO(path)
        if not hasattr(model, "names"):
            return None, f"{model_name} loaded but missing names"
        return model, None
    except Exception as e:
        return None, f"{model_name} load failed: {e}"

# -----------------------------
# Ensure general model exists (Colab)
# -----------------------------
if not Path(GENERAL_MODEL_PATH).exists():
    try:
        download_yolov8n(GENERAL_MODEL_PATH)
    except Exception as e:
        print("Auto-download failed:", e)
        print("Please upload yolov8n.pt using the upload prompt below or set GENERAL_MODEL_PATH.")

# -----------------------------
# Load models (general + automatic helmet)
# -----------------------------
print("Loading general model:", GENERAL_MODEL_PATH)
general_model, err = try_load_yolo(GENERAL_MODEL_PATH, "General model")
if err:
    raise SystemExit(err)
print("✔ General model loaded.")

helmet_model = None
if HELMET_MODEL_AUTOPATH and HELMET_MODEL_AUTOPATH.exists():
    print("Found helmet model at", str(HELMET_MODEL_AUTOPATH), "— loading automatically.")
    helmet_model, err = try_load_yolo(str(HELMET_MODEL_AUTOPATH), "Helmet model")
    if err:
        print("Warning: helmet model failed to load:", err)
        helmet_model = None
    else:
        print("✔ Helmet model loaded from", str(HELMET_MODEL_AUTOPATH))
else:
    print("No helmet model found at", str(HELMET_MODEL_AUTOPATH), "- helmet detection disabled (no prompt).")

# -----------------------------
# FILE UPLOAD: video or image (Colab UI)
# -----------------------------
print("Upload your image or video now (Colab upload dialog).")
uploaded_media = files.upload()
if not uploaded_media:
    print("No media uploaded. Exiting.")
else:
    media_name = list(uploaded_media.keys())[0]
    print("✔ Uploaded media:", media_name)

    input_path = media_name
    is_image = media_name.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp"))

    # -----------------------------
    # Tracker state (module level)
    # -----------------------------
    TRACKS = {}        # track_id -> { 'centroid': (x,y), 'frame': int, 'bbox': [x1,y1,x2,y2], 'speed_kmph': float }
    NEXT_TRACK_ID = 1

    def centroid_from_bbox(bbox):
        x1,y1,x2,y2 = bbox
        return ((x1+x2)/2.0, (y1+y2)/2.0)

    def euclidean(a,b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    def match_detections_to_tracks(detections_centroids, existing_tracks, max_dist=MAX_MATCH_DISTANCE_PX):
        matches = []
        used_tracks = set()
        for i, det_c in enumerate(detections_centroids):
            best_tid = None
            best_d = None
            for tid, t in existing_tracks.items():
                if tid in used_tracks:
                    continue
                d = euclidean(det_c, t['centroid'])
                if d <= max_dist and (best_d is None or d < best_d):
                    best_d = d
                    best_tid = tid
            if best_tid is not None:
                matches.append((i, best_tid))
                used_tracks.add(best_tid)
            else:
                matches.append((i, None))
        unmatched_tracks = [tid for tid in existing_tracks.keys() if tid not in used_tracks]
        return matches, unmatched_tracks

    def compute_speed_from_disp_pixels(dist_px, frame_delta, fps, px_per_meter=PX_PER_METER):
        if frame_delta <= 0 or fps <= 0:
            return 0.0
        dist_m = dist_px / px_per_meter
        time_s = frame_delta / fps
        if time_s == 0:
            return 0.0
        speed_m_s = dist_m / time_s
        return speed_m_s * 3.6  # km/h

    def density_from_count(count):
        if count <= density_thresholds["none"]:
            return "none"
        if count <= density_thresholds["low"]:
            return "low"
        if count <= density_thresholds["medium"]:
            return "medium"
        return "high"

    # -----------------------------
    # PROCESS IMAGE
    # -----------------------------
    def process_image_colab(file_path, camera_id="camera_0"):
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError("cv2.imread returned None. Check the file path and image file.")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_records = []

        gen_res = general_model(img)[0]

        helm_res = None
        if helmet_model is not None:
            try:
                helm_res = helmet_model(img)[0]
            except Exception as e:
                print("Warning: helmet model inference failed on image:", e)
                helm_res = None

        out_img = OUTPUT_DIR / f"annotated_{timestamp}.jpg"
        file_name = os.path.basename(file_path)

        # crowd count (people) in this single frame
        crowd_count = 0
        for b in gen_res.boxes:
            cls = int(b.cls[0])
            name = general_model.names[cls]
            if name.lower() == "person":
                crowd_count += 1
        density_level = density_from_count(crowd_count)

        helmet_boxes = []
        if helm_res is not None:
            for hb in helm_res.boxes:
                helmet_boxes.append({
                    "bbox": hb.xyxy[0].cpu().numpy().tolist(),
                    "cls": int(hb.cls[0]),
                    "conf": float(hb.conf[0])
                })

        for box in gen_res.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls  = int(box.cls[0])
            label = general_model.names[cls]
            color_rgb = mean_color_hex(img, xyxy)
            draw_box(img, xyxy, label, conf, (255,255,0))

            helmet_violation = False
            if helmet_boxes:
                x1,y1,x2,y2 = xyxy
                for hb in helmet_boxes:
                    hx1,hy1,hx2,hy2 = hb["bbox"]
                    inter_x = max(0, min(x2, hx2) - max(x1, hx1))
                    inter_y = max(0, min(y2, hy2) - max(y1, hy1))
                    if inter_x > 0 and inter_y > 0:
                        if hb["cls"] == 1:
                            helmet_violation = True
                            break
                        elif hb["cls"] == 0:
                            helmet_violation = False
                            break

            vehicle_id = make_vehicle_id()
            violation_type = ""
            violation_image = ""
            if helmet_violation:
                violation_type = "helmet"
                violation_image = save_violation_crop(img, xyxy, prefix="helmet_violation")

            # lane assignment by bbox centroid x
            cx, cy = centroid_from_bbox(xyxy)
            img_w = img.shape[1]
            lane_width = img_w / float(num_lanes)
            lane_idx = int(cx // lane_width) + 1
            lane_idx = max(1, min(num_lanes, lane_idx))

            rec = build_record(
                file_name=file_name,
                camera_id=camera_id,
                frame_id=0,
                vehicle_id=vehicle_id,
                bbox=xyxy,
                label=label,
                conf=conf,
                color=color_rgb,
                lane=f"lane_{lane_idx}",
                speed_kmph=None,
                overspeed=False,
                helmet_violation=helmet_violation,
                violation_type=violation_type,
                violation_image=violation_image,
                meta={"detector":"yolov8", "source":"image"},
                crowd_count=crowd_count,
                density_level=density_level
            )
            results_records.append(rec)

        cv2.imwrite(str(out_img), img)

        csv_path = OUTPUT_DIR / f"results_{timestamp}.csv"
        with open(csv_path, "w", newline='', encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=CSV_HEADERS)
            writer.writeheader()
            for r in results_records:
                writer.writerow(r)

        out_json = OUTPUT_DIR / f"result_{timestamp}.json"
        with open(out_json, "w", encoding="utf-8") as jf:
            json.dump(results_records, jf, indent=2, default=str)

        print("✔ Image processed:", out_img)
        print("✔ CSV saved:", csv_path)
        print("✔ JSON saved:", out_json)
        return str(out_img), str(out_json), results_records

    # -----------------------------
    # PROCESS VIDEO
    # -----------------------------
    def process_video_colab(file_path, camera_id="camera_0"):
        global TRACKS, NEXT_TRACK_ID  # tracker state is module-level; use global here
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {file_path}")
        fps_local = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_vid  = OUTPUT_DIR / f"annotated_{timestamp}.mp4"
        out_json = OUTPUT_DIR / f"result_{timestamp}.json"
        csv_path = OUTPUT_DIR / f"results_{timestamp}.csv"

        writer = cv2.VideoWriter(str(out_vid), cv2.VideoWriter_fourcc(*"mp4v"), fps_local, (w,h))
        results_records = []
        frame_id = 0
        print("Processing video... FPS:", fps_local)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_id += 1

                gen_res = general_model(frame)[0]

                # crowd count (persons)
                crowd_count = 0
                for b in gen_res.boxes:
                    cls = int(b.cls[0])
                    name = general_model.names[cls]
                    if name.lower() == "person":
                        crowd_count += 1
                density_level = density_from_count(crowd_count)

                helm_res = None
                if helmet_model is not None:
                    try:
                        helm_res = helmet_model(frame)[0]
                    except Exception as e:
                        if frame_id == 1:
                            print("Warning: helmet model inference failing on video frames:", e)
                        helm_res = None

                helmet_boxes = []
                if helm_res is not None:
                    for hb in helm_res.boxes:
                        helmet_boxes.append({
                            "bbox": hb.xyxy[0].cpu().numpy().tolist(),
                            "cls": int(hb.cls[0]),
                            "conf": float(hb.conf[0])
                        })

                # Build current detections list
                detections = []
                for box in gen_res.boxes:
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = general_model.names[cls]
                    cx, cy = centroid_from_bbox(xyxy)
                    detections.append({"bbox": xyxy, "centroid": (cx,cy), "label": label, "conf": conf})

                # Match detections -> tracks
                det_centroids = [d['centroid'] for d in detections]
                matches, unmatched_tracks = match_detections_to_tracks(det_centroids, TRACKS, max_dist=MAX_MATCH_DISTANCE_PX)

                det_to_tid = {}
                for det_idx, tid in matches:
                    det_to_tid[det_idx] = tid  # tid may be None

                # Create/update tracks and compute speed
                for i, d in enumerate(detections):
                    tid = det_to_tid.get(i)
                    if tid is None:
                        # new track
                        tid = NEXT_TRACK_ID
                        NEXT_TRACK_ID += 1
                        TRACKS[tid] = {
                            "centroid": d["centroid"],
                            "frame": frame_id,
                            "bbox": d["bbox"],
                            "speed_kmph": 0.0
                        }
                    else:
                        prev = TRACKS[tid]
                        prev_cent = prev['centroid']
                        prev_frame = prev['frame']
                        dist_px = euclidean(prev_cent, d['centroid'])
                        frame_delta = frame_id - prev_frame
                        speed_kmph = compute_speed_from_disp_pixels(dist_px, frame_delta, fps_local, px_per_meter=PX_PER_METER)
                        smoothed = (1.0-SPEED_SMOOTH_ALPHA) * prev.get('speed_kmph', 0.0) + SPEED_SMOOTH_ALPHA * speed_kmph
                        TRACKS[tid].update({
                            "centroid": d['centroid'],
                            "frame": frame_id,
                            "bbox": d['bbox'],
                            "speed_kmph": smoothed
                        })
                    # attach track info to detection
                    d['track_id'] = tid
                    d['speed_kmph'] = TRACKS[tid]['speed_kmph']
                    d['overspeed'] = TRACKS[tid]['speed_kmph'] > OVERSPEED_THRESHOLD_KMPH

                # remove stale tracks
                stale_tids = [tid for tid, t in TRACKS.items() if (frame_id - t['frame']) > 150]
                for tid in stale_tids:
                    del TRACKS[tid]

                # Build records and draw
                for d in detections:
                    xyxy = d['bbox']
                    conf = d['conf']
                    label = d['label']

                    draw_box(frame, xyxy, label, conf, (255,255,0))

                    helmet_violation = False
                    if helmet_boxes:
                        x1,y1,x2,y2 = xyxy
                        for hb in helmet_boxes:
                            hx1,hy1,hx2,hy2 = hb["bbox"]
                            inter_x = max(0, min(x2, hx2) - max(x1, hx1))
                            inter_y = max(0, min(y2, hy2) - max(y1, hy1))
                            if inter_x > 0 and inter_y > 0:
                                if hb["cls"] == 1:
                                    helmet_violation = True
                                    break
                                elif hb["cls"] == 0:
                                    helmet_violation = False
                                    break

                    vehicle_id = make_vehicle_id()
                    violation_type = ""
                    violation_image = ""
                    if helmet_violation:
                        violation_type = "helmet"
                        violation_image = save_violation_crop(frame, xyxy, prefix="helmet_violation")

                    # lane assignment
                    cx, cy = d['centroid']
                    img_w = frame.shape[1]
                    lane_width = img_w / float(num_lanes)
                    lane_idx = int(cx // lane_width) + 1
                    lane_idx = max(1, min(num_lanes, lane_idx))

                    color_hex = mean_color_hex(frame, xyxy)

                    rec = build_record(
                        file_name=os.path.basename(file_path),
                        camera_id=camera_id,
                        frame_id=frame_id,
                        vehicle_id=vehicle_id,
                        bbox=xyxy,
                        label=label,
                        conf=conf,
                        color=color_hex,
                        lane=f"lane_{lane_idx}",
                        speed_kmph=d.get('speed_kmph', 0.0),
                        overspeed=d.get('overspeed', False),
                        helmet_violation=helmet_violation,
                        violation_type=violation_type,
                        violation_image=violation_image,
                        meta={"detector":"yolov8", "source":"video", "track_id": d.get('track_id')},
                        crowd_count=crowd_count,
                        density_level=density_level
                    )
                    results_records.append(rec)

                writer.write(frame)
                if frame_id % 50 == 0:
                    print(f"Processed {frame_id} frames... crowd_count={crowd_count} density={density_level}")
        finally:
            cap.release()
            writer.release()

        # write outputs
        with open(csv_path, "w", newline='', encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=CSV_HEADERS)
            writer.writeheader()
            for r in results_records:
                writer.writerow(r)

        with open(out_json, "w", encoding="utf-8") as jf:
            json.dump(results_records, jf, indent=2, default=str)

        print("✔ Video processed:", out_vid)
        print("✔ CSV saved:", csv_path)
        print("✔ JSON saved:", out_json)

        return str(out_vid), str(out_json), results_records

    # -----------------------------
    # Run appropriate processor
    # -----------------------------
    try:
        if is_image:
            ann, js, recs = process_image_colab(input_path)
            files.download(ann)
            files.download(js)
            csv_fp = list(OUTPUT_DIR.glob("results_*.csv"))[-1]
            files.download(str(csv_fp))
        else:
            ann, js, recs = process_video_colab(input_path)
            files.download(ann)
            files.download(js)
            csv_fp = list(OUTPUT_DIR.glob("results_*.csv"))[-1]
            files.download(str(csv_fp))
        print("DONE — outputs saved to /content/outputs")
    except Exception as e:
        print("Processing failed:", e)
        raise
