# debug_detect_fire_smoke.py
import cv2
import numpy as np
import sys
import time

# ---------- EMAIL CONFIG ----------
import smtplib
from email.message import EmailMessage

EMAIL_USER = "t39775496@gmail.com"          # sender + receiver
EMAIL_TO   = "t39775496@gmail.com"          # email to receive alerts
EMAIL_PASS = "uvnzoepxrqnxrwqn"             # your 16-letter app password

EMAIL_COOLDOWN = 60  # seconds between video alerts


def send_email_alert(subject, body, image_path=None):
    msg = EmailMessage()    
    msg["From"] = EMAIL_USER
    msg["To"] = EMAIL_TO
    msg["Subject"] = subject
    msg.set_content(body)

    # attach image if available
    if image_path is not None:
        try:
            with open(image_path, "rb") as f:
                data = f.read()
            msg.add_attachment(
                data,
                maintype="image",
                subtype="jpeg",
                filename="alert.jpg"
            )
        except Exception as e:
            print("⚠️ Could not attach image:", e)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_USER, EMAIL_PASS)
            smtp.send_message(msg)
        print("📨 Email alert sent!")
        return True
    except Exception as e:
        print("❌ Email failed:", e)
        return False


# ---------- COLORS ----------
FIRE_COLOR = (0, 0, 255)    # RED
SMOKE_COLOR = (255, 0, 0)   # BLUE


# ---------- IMAGE MODE ----------
def classify_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("❌ Image not found:", img_path)
        return

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    fire_mask = cv2.inRange(hsv, (0, 80, 80), (50, 255, 255))
    smoke_mask = cv2.inRange(hsv, (0, 0, 60), (180, 70, 230))

    fire_pixels = cv2.countNonZero(fire_mask)
    smoke_pixels = cv2.countNonZero(smoke_mask)

    MIN_FIRE_PIXELS = 500

    if fire_pixels < MIN_FIRE_PIXELS and smoke_pixels < MIN_FIRE_PIXELS:
        print("No Fire or Smoke detected")

    # show image
        cv2.imshow("Result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # send email
        snap = "no_fire_alert.jpg"
        cv2.imwrite(snap, img)

        subject = "✅ NO FIRE OR SMOKE DETECTED"
        body = "System checked the image and no fire or smoke was detected."

        send_email_alert(subject, body, snap)

        return

    total = max(fire_pixels + smoke_pixels, 1)
    fire_pct = fire_pixels * 100.0 / total
    smoke_pct = smoke_pixels * 100.0 / total
    if fire_pct < 5:
        print("No Fire or Smoke detected")
        snap = "no_fire_alert.jpg"
        cv2.imwrite(snap, img)
        subject = "✅ NO FIRE OR SMOKE DETECTED"
        body = "System checked the image and no fire or smoke was detected."
        send_email_alert(subject, body, snap)
        return
    print(f"\n🔥 FIRE detected:  {fire_pct:.1f}%")
    print(f"💨 SMOKE detected: {smoke_pct:.1f}%\n")

    # -------- EMAIL ALERT FOR IMAGE MODE --------
    snap = "image_alert.jpg"
    cv2.imwrite(snap, img)

    subject = "🔥 FIRE ALERT (IMAGE)" if fire_pct > smoke_pct else "💨 SMOKE ALERT (IMAGE)"
    body = f"Image detection:\nFire: {fire_pct:.1f}%\nSmoke: {smoke_pct:.1f}%"

    send_email_alert(subject, body, snap)
    print("📨 Email alert sent for image!")


# If an image path is provided → ONLY run image mode
if len(sys.argv) == 2:
    classify_image(sys.argv[1])
    sys.exit()


# ---------- VIDEO MODE ----------
VIDEO = "fire2.mp4"
OUTPUT = "debug_output_fire_detection.mp4"

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    print("❌ Could not open video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(OUTPUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

print("Running... Press q to quit.")


def get_largest_box(mask, min_area=500):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(largest)
    if bw * bh < min_area:
        return None
    return (x, y, bw, bh)


last_email_time = 0


while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    fire_mask = cv2.inRange(hsv, (0, 80, 80), (50, 255, 255))
    smoke_mask = cv2.inRange(hsv, (0, 0, 60), (180, 70, 230))
    fire_pixels = cv2.countNonZero(fire_mask)
    smoke_pixels = cv2.countNonZero(smoke_mask)

    MIN_FIRE_PIXELS = 500

    if fire_pixels < MIN_FIRE_PIXELS:

        cv2.putText(frame,
                "NO FIRE OR SMOKE DETECTED",
                (50,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2)

        cv2.imshow("Dynamic Fire + Smoke Boxes", frame)
        out.write(frame)
        continue

    total = max(fire_pixels + smoke_pixels, 1)
    fire_pct = fire_pixels * 100.0 / total
    smoke_pct = smoke_pixels * 100.0 / total

    now = time.time()

    # ---------- FIRE BOX ----------
    fire_box = get_largest_box(fire_mask)
    if fire_box is not None:
        x, y, bw, bh = fire_box
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), FIRE_COLOR, 3)
        cv2.putText(frame, f"FIRE {fire_pct:.1f}%", (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, FIRE_COLOR, 2)

        if now - last_email_time > EMAIL_COOLDOWN:
            snap = "fire_alert.jpg"
            cv2.imwrite(snap, frame)
            if send_email_alert("🔥 FIRE ALERT", f"Fire detected: {fire_pct:.1f}%", snap):
                last_email_time = now

    # ---------- SMOKE BOX ----------
    smoke_box = get_largest_box(smoke_mask)
    if smoke_box is not None:
        sx, sy, sbw, sbh = smoke_box
        cv2.rectangle(frame, (sx, sy), (sx + sbw, sy + sbh), SMOKE_COLOR, 3)
        cv2.putText(frame, f"SMOKE {smoke_pct:.1f}%", (sx, max(20, sy - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, SMOKE_COLOR, 2)

        if now - last_email_time > EMAIL_COOLDOWN:
            snap = "smoke_alert.jpg"
            cv2.imwrite(snap, frame)
            if send_email_alert("💨 SMOKE ALERT", f"Smoke detected: {smoke_pct:.1f}%", snap):
                last_email_time = now

    # ---------- SHOW VIDEO ----------
    cv2.imshow("Dynamic Fire + Smoke Boxes", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
out.release()
cv2.destroyAllWindows()
print("Saved ->", OUTPUT)


