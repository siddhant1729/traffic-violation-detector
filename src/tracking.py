from ultralytics import YOLO
import cv2
import math

model = YOLO("yolov8n.pt")

# Classes we care about
VEHICLE_CLASSES = ["car", "motorbike", "bus", "truck"]

# To store object ID and centroid
tracked_objects = {}
object_id = 0  # ✅ define once here

def get_center(x1, y1, x2, y2):
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

cap = cv2.VideoCapture("data/sample_video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    detections = []

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        if label in VEHICLE_CLASSES:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append((x1, y1, x2, y2))

    current_objects = {}

    # Compare new detections to existing tracked ones
    for (x1, y1, x2, y2) in detections:
        cx, cy = get_center(x1, y1, x2, y2)

        same_object_detected = False
        for id, pt in tracked_objects.items():
            dist = math.hypot(cx - pt[0], cy - pt[1])
            if dist < 35:  # Threshold for same object
                current_objects[id] = (cx, cy)
                same_object_detected = True
                break

        if not same_object_detected:
            # ❌ Don't use 'global' here — just increment normally
            object_id += 1
            current_objects[object_id] = (cx, cy)

    # Update tracked objects
    tracked_objects = current_objects.copy()

    # Draw boxes + IDs
    for id, pt in tracked_objects.items():
        cv2.circle(frame, pt, 4, (0, 255, 0), -1)
        cv2.putText(frame, f"ID {id}", (pt[0] - 10, pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Vehicle Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
