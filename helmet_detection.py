import cv2
from ultralytics import YOLO
import os

model = YOLO("helmet.pt")

if not os.path.exists("output"):
    os.makedirs("output")

img = cv2.imread("test2.jpg")

if img is None:

    print("Image not found!")
    exit()


results = model(img)

for r in results:

    for box in r.boxes:

       
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        
        conf = float(box.conf[0])

      
        if conf < 0.5:
            continue

      
        head_x1 = x1
        head_y1 = y1

        head_x2 = x2
        head_y2 = y1 + (y2 - y1) // 4

        if conf > 0.7:

            label = "Helmet"
            color = (0, 255, 0)

        else:

            label = "No Helmet"
            color = (0, 0, 255)

        
        cv2.rectangle(
            img,
            (head_x1, head_y1),
            (head_x2, head_y2),
            color,
            3
        )

       
        cv2.putText(
            img,
            f"{label} {conf:.2f}",
            (head_x1, head_y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )


cv2.imwrite("output/result.jpg", img)


cv2.imshow("Helmet Detection", img)

cv2.waitKey(0)

cv2.destroyAllWindows()

print("Detection completed.")
