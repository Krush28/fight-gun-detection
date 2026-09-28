from ultralytics import YOLO
import cv2

# YOLO Pose model
MODEL_PATH = "model/yolo/yolov8n-pose.pt"

# Load model
model = YOLO(MODEL_PATH)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("YOLO + ByteTrack started.")
print("Press Q to quit.")

while True:

    success, frame = cap.read()

    if not success:
        print("Failed to read frame.")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # --------------------------------------------------
    # YOLO + BYTE TRACK
    # --------------------------------------------------

    results = model.track(
        source=frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],          # 0 = person
        conf=0.5,
        verbose=False
    )

    result = results[0]

    # --------------------------------------------------
    # DRAW DETECTIONS
    # --------------------------------------------------

    annotated_frame = result.plot()

    # --------------------------------------------------
    # GET TRACKING IDs
    # --------------------------------------------------

    if result.boxes is not None:

        boxes = result.boxes

        if boxes.id is not None:

            track_ids = boxes.id.int().cpu().tolist()

            for box, track_id in zip(
                boxes.xyxy.cpu().tolist(),
                track_ids
            ):

                x1, y1, x2, y2 = map(int, box)

                # Person ID
                label = f"Person ID: {track_id}"

                # Draw ID
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, max(y1 - 10, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                # Draw ID box
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 255),
                    2
                )

    # --------------------------------------------------
    # DISPLAY
    # --------------------------------------------------

    cv2.imshow(
        "YOLO + ByteTrack - Person Tracking",
        annotated_frame
    )

    # Quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ------------------------------------------------------
# CLEANUP
# ------------------------------------------------------

cap.release()
cv2.destroyAllWindows()

print("Tracking stopped.")