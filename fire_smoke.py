from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")  
images = ["fire.jpg"]     
COLORS = {"fire": (0, 0, 255), "smoke": (255, 0, 0)}
CONF_THRESH = 0.30

for image_file in images:
    img = cv2.imread(image_file)
    if img is None:
        print(f"Error: {image_file} not found!")
        continue

    results = model(img)
    r = results[0]
    detected = False

    if r.boxes is not None and len(r.boxes) > 0:
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        for box, cls_id, conf in zip(xyxy, cls_ids, confs):
            if conf < CONF_THRESH:
                continue
            label = r.names[int(cls_id)].lower()
            if label in ["fire", "smoke"]:
                detected = True
                x1, y1, x2, y2 = map(lambda v: int(round(v)), box)
                color = COLORS.get(label, (0, 255, 0))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f"{label.upper()} {conf*100:.1f}%",
                            (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if detected:
        print(f"Image detected: {image_file}")
    else:
        print(f"No detections in image: {image_file}")

    cv2.imshow(image_file, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
