from ultralytics import YOLO
import cv2

MODEL_PATH = "model/gun/best.pt"

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, conf=0.40, verbose=False)

    annotated = results[0].plot()

    cv2.imshow("Gun Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()