from ultralytics import YOLO
import cv2
import math
import time
from datetime import datetime
from database import init_db, log_violation

# Load YOLO model
model = YOLO("yolov8n.pt")

# Vehicle types we care about
VEHICLE_CLASSES = ["car", "motorbike", "bus", "truck"]

# Data structures
tracked_objects = {}
previous_positions = {}
violations = set()
object_id = 0

# Stop line position (y-coordinate)
STOP_LINE_Y = 300  # adjust based on your video
LINE_THICKNESS = 3

# Light state simulation
LIGHT_STATE = "RED"  # can be toggled to GREEN for testing

cap = cv2.VideoCapture("data/sample_video.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS: {fps}")

def get_center(x1, y1, x2, y2):
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

# Initialize the SQLite database
init_db()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for smoother processing
    frame = cv2.resize(frame, (800, 450))

    # Draw stop line
    line_color = (0, 0, 255) if LIGHT_STATE == "RED" else (0, 255, 0)
    cv2.line(frame, (0, STOP_LINE_Y), (frame.shape[1], STOP_LINE_Y), line_color, LINE_THICKNESS)

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

    # Check for violations
    for id, pt in current_objects.items():
        cx, cy = pt

        if LIGHT_STATE == "RED" and STOP_LINE_Y - 10 < cy < STOP_LINE_Y + 10:
            if id not in violations:
                violations.add(id)
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"logs/redlight_{id}_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"🚨 Violation detected! Vehicle ID {id} crossed red at {timestamp}")

                # ✅ Log the violation into SQLite database
                log_violation(id, "Red Light", filename)

        # Draw ID on vehicles
        color = (0, 0, 255) if id in violations else (0, 255, 0)
        cv2.circle(frame, (cx, cy), 4, color, -1)
        cv2.putText(frame, f"ID {id}", (cx - 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    tracked_objects = current_objects.copy()

    # Display light status
    cv2.putText(frame, f"LIGHT: {LIGHT_STATE}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, line_color, 2)

    cv2.imshow("Red Light Violation Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
