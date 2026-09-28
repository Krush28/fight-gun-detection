from ultralytics import YOLO
import cv2

model = YOLO("model/gun/best.pt")

image = cv2.imread("test_gun.jpg")

results = model(image, conf=0.40)

result = results[0]

print("Detected classes:")

for box in result.boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])

    print(
        model.names[class_id],
        "Confidence:",
        round(confidence, 2)
    )

annotated = result.plot()

cv2.imshow("Gun Detection", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()