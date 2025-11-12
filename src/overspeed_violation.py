from ultralytics import YOLO
import cv2
import math
from joblib import load
from datetime import datetime
from database import init_db, log_violation

# --- Load YOLO and ML model ---
model = YOLO("yolov8n.pt")
rf_model = load("ml_models/overspeed_rf.pkl")

VEHICLE_CLASSES = ["car", "motorbike", "bus", "truck"]
tracked_objects = {}
previous_positions = {}
object_id = 0
violations = set()

# --- Constants ---
PIXELS_PER_METER = 8.0
STOP_LINE_Y = 300
LINE_THICKNESS = 2
SPEED_LIMIT = 60  # fallback limit (km/h)

cap = cv2.VideoCapture("data/sample_video.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS: {fps}")

init_db()

def get_center(x1, y1, x2, y2):
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (800, 450))
    results = model(frame, verbose=False)

    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        if label in VEHICLE_CLASSES:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append((x1, y1, x2, y2))

    current_objects = {}
    for (x1, y1, x2, y2) in detections:
        cx, cy = get_center(x1, y1, x2, y2)
        same_object_detected = False
        for id, pt in tracked_objects.items():
            dist = math.hypot(cx - pt[0], cy - pt[1])
            if dist < 35:
                current_objects[id] = (cx, cy)
                same_object_detected = True
                break
        if not same_object_detected:
            object_id += 1
            current_objects[object_id] = (cx, cy)

    # --- ML-based speed prediction ---
    for id, new_pt in current_objects.items():
        if id in previous_positions:
            prev_pt = previous_positions[id]
            dist_pixels = math.hypot(new_pt[0] - prev_pt[0], new_pt[1] - prev_pt[1])
            dist_meters = dist_pixels / PIXELS_PER_METER
            speed_m_per_s = dist_meters * fps
            speed_kmh = speed_m_per_s * 3.6

            # --- Predict with Random Forest ---
            features = [[dist_pixels, fps, PIXELS_PER_METER]]
            prediction = rf_model.predict(features)[0]

            color = (0, 255, 0)
            if prediction == "Overspeed" or speed_kmh > SPEED_LIMIT:
                if id not in violations:
                    violations.add(id)
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    filename = f"logs/overspeed_{id}_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    log_violation(id, "Overspeed", filename)
                    print(f"🚨 Overspeed Detected | Vehicle {id} | {int(speed_kmh)} km/h")
                color = (0, 0, 255)

            cv2.putText(frame, f"ID {id} | {int(speed_kmh)} km/h",
                        (new_pt[0] - 20, new_pt[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            cv2.putText(frame, f"ID {id}", (new_pt[0] - 10, new_pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    previous_positions = current_objects.copy()
    tracked_objects = current_objects.copy()

    cv2.imshow("Overspeed Detection (ML)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
