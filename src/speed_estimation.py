from ultralytics import YOLO
import cv2
import math
import time
from collections import defaultdict

# Load YOLOv8 model (Nano for fastest speed)
model = YOLO("yolov8n.pt")

# Vehicle classes YOLO can detect
VEHICLE_CLASSES = ["car", "motorbike", "bus", "truck"]

# Tracking data structures
tracked_objects = {}
previous_positions = {}
speed_history = defaultdict(list)
object_id = 0

# Conversion constant (tune based on video)
PIXELS_PER_METER = 9.0

# Load video
cap = cv2.VideoCapture("data/sample_video.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video FPS: {fps}")

# Resize for smoother inference
resize_width, resize_height = 640, 360

# FPS monitoring
prev_time = time.time()

def get_center(x1, y1, x2, y2):
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (resize_width, resize_height))
    frame_count += 1

    # Skip alternate frames for better performance
    if frame_count % 2 != 0:
        continue

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

    # Speed Calculation
    for id, new_pt in current_objects.items():
        if id in previous_positions:
            prev_pt = previous_positions[id]
            dist_pixels = math.hypot(new_pt[0] - prev_pt[0], new_pt[1] - prev_pt[1])

            # Ignore unrealistic jumps
            if 1 < dist_pixels < 100:
                dist_meters = dist_pixels / PIXELS_PER_METER
                speed_m_per_s = dist_meters * fps
                speed_kmh = speed_m_per_s * 3.6

                if 0.5 < speed_kmh < 200:
                    # Smooth speed using moving average
                    speed_history[id].append(speed_kmh)
                    if len(speed_history[id]) > 5:
                        speed_history[id].pop(0)
                    avg_speed = sum(speed_history[id]) / len(speed_history[id])

                    cv2.putText(frame, f"ID {id} | {int(avg_speed)} km/h",
                                (new_pt[0] - 20, new_pt[1] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(frame, f"ID {id}", (new_pt[0] - 10, new_pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    previous_positions = current_objects.copy()
    tracked_objects = current_objects.copy()

    # FPS display
    curr_time = time.time()
    fps_live = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps_live)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Smooth Speed Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
