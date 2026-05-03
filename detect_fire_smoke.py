from ultralytics import YOLO
import cv2
import time
import os
import smtplib
from email.message import EmailMessage
import mimetypes

MODEL = "yolov8n.pt"  
VIDEO = "fire2.mp4"
OUTPUT = "debug_output_fire_detection.mp4"


CONF_THRESH = 0.25


FALLBACK_COLOR = (0, 255, 0)
PALETTE = {
    "fire": (0, 0, 255),
    "smoke": (255, 0, 0)
}


EMAIL_COOLDOWN = 60


def send_email_alert(subject: str, body: str, attachment_path: str = None):
    """
    Sends an email using environment variables:
      ALERT_EMAIL_USER, ALERT_EMAIL_PASS, ALERT_EMAIL_TO
    Returns True on success, False otherwise.
    """
    user = os.getenv("ALERT_EMAIL_USER")
    passwd = os.getenv("ALERT_EMAIL_PASS")
    to_addr = os.getenv("ALERT_EMAIL_TO")

    if not (user and passwd and to_addr):
        print("❌ Email vars not set. ALERT_EMAIL_USER, PASS, TO missing.")
        return False

    msg = EmailMessage()
    msg["From"] = user
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content(body)


    if attachment_path:
        if os.path.exists(attachment_path):
            ctype, encoding = mimetypes.guess_type(attachment_path)
            if ctype is None:
                ctype = "application/octet-stream"
            maintype, subtype = ctype.split("/", 1)

            with open(attachment_path, "rb") as f:
                msg.add_attachment(
                    f.read(),
                    maintype=maintype,
                    subtype=subtype,
                    filename=os.path.basename(attachment_path)
                )
        else:
            print("⚠️ Attachment not found:", attachment_path)

    try:
   
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(user, passwd)
            smtp.send_message(msg)

        print("📨 Email alert sent to", to_addr)
        return True

    except Exception as e:
        print("❌ Failed to send email:", e)
        return False


model = YOLO(MODEL)
cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    print("❌ Could not open video:", VIDEO)
    exit()

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(OUTPUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

print("DEBUG MODE — drawing all detections. Press q to quit.")
frame_idx = 0
last_print = 0.0
last_email_sent = 0.0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    results = model(frame)
    r = results[0]

    labels_this_frame = []
    if r.boxes is not None and len(r.boxes) > 0:
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        for box, cid, conf in zip(xyxy, cls_ids, confs):
            if conf < CONF_THRESH:
                continue

            label = r.names[int(cid)].lower()
            labels_this_frame.append((label, float(conf)))

            x1, y1, x2, y2 = map(lambda v: int(round(v)), box)
            color = PALETTE.get(label, FALLBACK_COLOR)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label.upper()} {conf*100:.1f}%",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    
    now = time.time()
    if labels_this_frame and now - last_print > 0.5:
        last_print = now
        pretty = ", ".join([f"{lab}:{conf*100:.1f}%" for lab,conf in labels_this_frame])
        print(f"frame {frame_idx}: {pretty}")

  
    if labels_this_frame and (time.time() - last_email_sent) > EMAIL_COOLDOWN:
       
        snap_path = f"alert_frame_{frame_idx}.jpg"
        cv2.imwrite(snap_path, frame)

       
       
        top_label, top_conf = max(labels_this_frame, key=lambda x: x[1])
        subject = f"ALERT: {top_label.upper()} detected"
        body = f"Detected {top_label} with confidence {top_conf*100:.1f}% in video {VIDEO} at frame {frame_idx}. See attached image."


        sent = send_email_alert(subject, body, attachment_path=snap_path)
        if sent:
            last_email_sent = time.time()

    cv2.imshow("DEBUG Fire/Smoke Detection", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Saved ->", OUTPUT)


def check_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("❌ Image not found!")
        return

    results = model(img)[0]

    if results.boxes is not None and len(results.boxes) > 0:
        confs = results.boxes.conf.cpu().numpy()
        if np.any(confs >= CONF_THRESH):
            print("Image detected")
            return

    print("No detection")


check_image("fire.jpg")
