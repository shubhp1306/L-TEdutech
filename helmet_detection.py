import cv2
from ultralytics import YOLO
import os

# Load model
model = YOLO("helmet.pt")

# Create output folder
if not os.path.exists("output"):
    os.makedirs("output")

# Read image
img = cv2.imread("test.jpeg")

# Run detection
results = model(img)

# Process detections
for r in results:

    for box in r.boxes:

        # Coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Confidence
        conf = float(box.conf[0])

        # Ignore weak detections
        if conf < 0.5:
            continue

        # =========================
        # HEAD REGION ONLY
        # =========================

        head_x1 = x1
        head_y1 = y1

        head_x2 = x2
        head_y2 = y1 + (y2 - y1) // 4

        # Simulated helmet logic
        if conf > 0.7:

            label = "Helmet"
            color = (0, 255, 0)

        else:

            label = "No Helmet"
            color = (0, 0, 255)

        # Draw ONLY head box
        cv2.rectangle(img,
                      (head_x1, head_y1),
                      (head_x2, head_y2),
                      color,
                      3)

        # Label
        cv2.putText(img,
                    label,
                    (head_x1, head_y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2)

# Save output
cv2.imwrite("output/result.jpg", img)

# Show result
cv2.imshow("Helmet Detection", img)

cv2.waitKey(0)

cv2.destroyAllWindows()