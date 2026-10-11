import cv2
import csv
import os
from ultralytics import YOLO

# ==========================================
# 1. PROJECT SETTINGS
# ==========================================

VIDEO_PATH = "video/cars.mp4"
OUTPUT_PATH = "output/annotated_cars.mp4"
CSV_PATH = "annotation/car_annotations.csv"

# Create output folders if they do not exist
os.makedirs("output", exist_ok=True)
os.makedirs("annotation", exist_ok=True)

# Vehicle classes in the COCO dataset:
# 2 = Car
# 3 = Motorcycle
# 5 = Bus
# 7 = Truck

VEHICLE_CLASSES = [2, 3, 5, 7]

# ==========================================
# 2. LOAD YOLO MODEL
# ==========================================

print("Loading YOLO model...")

# Use yolo11s.pt for faster processing.
# You can try yolo11m.pt for potentially better detection.
model = YOLO("yolo11s.pt")

print("YOLO model loaded successfully!")

# ==========================================
# 3. OPEN INPUT VIDEO
# ==========================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video:", VIDEO_PATH)
    raise SystemExit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if fps <= 0:
    fps = 24.0

print(f"Video resolution: {width} x {height}")
print(f"FPS: {fps}")
print(f"Total frames: {total_frames}")

# ==========================================
# 4. CREATE OUTPUT VIDEO
# ==========================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)

if not out.isOpened():
    print("ERROR: Could not create output video.")
    cap.release()
    raise SystemExit(1)

# ==========================================
# 5. ANNOTATION SETTINGS
# ==========================================

# Red in OpenCV BGR format
RED = (0, 0, 255)

# White text
WHITE = (255, 255, 255)

# Thick bounding box
BOX_THICKNESS = 5

# Large label text
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.8
FONT_THICKNESS = 4

# Large label bar padding
PADDING_X = 20
PADDING_Y = 18

# Detection settings
CONFIDENCE = 0.10
IMAGE_SIZE = 1280
MAX_DETECTIONS = 1000

# ==========================================
# 6. INITIALIZE TRACKING COUNTERS
# ==========================================

frame_count = 0
total_detections = 0
unique_track_ids = set()

# ==========================================
# 7. CREATE CSV ANNOTATION FILE
# ==========================================

csv_file = open(
    CSV_PATH,
    mode="w",
    newline="",
    encoding="utf-8"
)

csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "Frame",
    "Timestamp_seconds",
    "Track_ID",
    "Vehicle_Class",
    "Confidence",
    "X1",
    "Y1",
    "X2",
    "Y2"
])

# ==========================================
# 8. PROCESS VIDEO FRAME BY FRAME
# ==========================================

print("\nStarting vehicle detection and annotation...\n")

try:
    while True:

        # Read the next video frame
        success, frame = cap.read()

        if not success:
            break

        frame_count += 1

        # Calculate video timestamp
        timestamp = frame_count / fps

        # --------------------------------------
        # RUN YOLO VEHICLE DETECTION AND TRACKING
        # --------------------------------------

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=VEHICLE_CLASSES,
            conf=CONFIDENCE,
            iou=0.5,
            imgsz=IMAGE_SIZE,
            max_det=MAX_DETECTIONS,
            verbose=False
        )

        # --------------------------------------
        # DRAW VEHICLE BOXES AND ID LABELS
        # --------------------------------------

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                # Get bounding box coordinates
                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                # Keep coordinates within the video frame
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))

                # Get vehicle class
                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                # Get confidence score
                confidence = float(box.conf[0])

                # Get tracking ID
                if box.id is not None:
                    track_id = int(box.id[0])
                    unique_track_ids.add(track_id)
                    track_text = str(track_id)
                else:
                    track_id = -1
                    track_text = "N/A"

                # Count detections
                total_detections += 1

                # ----------------------------------
                # DRAW THICK RED VEHICLE BOX
                # ----------------------------------

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    RED,
                    BOX_THICKNESS
                )

                # ----------------------------------
                # CREATE LARGE ID LABEL
                # ----------------------------------

                label = (
                    f"ID: {track_text} | "
                    f"{class_name.upper()} | "
                    f"{confidence:.2f}"
                )

                # Measure label text dimensions
                (text_width, text_height), baseline = cv2.getTextSize(
                    label,
                    FONT,
                    FONT_SCALE,
                    FONT_THICKNESS
                )

                # Calculate label bar size
                bar_width = text_width + (PADDING_X * 2)
                bar_height = (
                    text_height
                    + baseline
                    + (PADDING_Y * 2)
                )

                # Prefer placing the label above the box
                label_top = y1 - bar_height

                # If there is insufficient space above,
                # place the label inside the top of the box
                if label_top < 0:
                    label_top = y1

                # Keep label bar within frame boundaries
                label_top = min(
                    label_top,
                    height - bar_height
                )
                label_top = max(0, label_top)

                label_right = min(
                    width - 1,
                    x1 + bar_width
                )

                label_bottom = min(
                    height - 1,
                    label_top + bar_height
                )

                # Draw filled red ID label bar
                cv2.rectangle(
                    frame,
                    (x1, label_top),
                    (label_right, label_bottom),
                    RED,
                    -1
                )

                # Calculate white text position
                text_x = x1 + PADDING_X
                text_y = (
                    label_top
                    + PADDING_Y
                    + text_height
                )

                # Ensure text fits inside the frame
                text_x = min(
                    text_x,
                    max(0, width - text_width - 2)
                )

                # Draw large white ID label text
                cv2.putText(
                    frame,
                    label,
                    (text_x, text_y),
                    FONT,
                    FONT_SCALE,
                    WHITE,
                    FONT_THICKNESS,
                    cv2.LINE_AA
                )

                # ----------------------------------
                # SAVE ANNOTATION TO CSV
                # ----------------------------------

                csv_writer.writerow([
                    frame_count,
                    round(timestamp, 3),
                    track_id,
                    class_name,
                    round(confidence, 4),
                    x1,
                    y1,
                    x2,
                    y2
                ])

        # --------------------------------------
        # SAVE ANNOTATED FRAME TO OUTPUT VIDEO
        # --------------------------------------

        out.write(frame)

        # Show progress every 25 frames
        if frame_count % 25 == 0:
            print(
                f"Processed {frame_count}/{total_frames} frames | "
                f"Detections: {total_detections} | "
                f"Unique IDs: {len(unique_track_ids)}"
            )

finally:

    # ==========================================
    # 9. RELEASE RESOURCES
    # ==========================================

    cap.release()
    out.release()
    csv_file.close()

# ==========================================
# 10. FINAL SUMMARY
# ==========================================

print("\n" + "=" * 50)
print("ANNOTATION COMPLETED!")
print("=" * 50)

print(f"Frames processed: {frame_count}")
print(f"Vehicle detections: {total_detections}")
print(f"Unique tracked IDs: {len(unique_track_ids)}")

print(f"\nAnnotated video: {OUTPUT_PATH}")
print(f"CSV annotations: {CSV_PATH}")

print("\nOpen the annotated video using:")
print(f'Invoke-Item "{OUTPUT_PATH}"')