import subprocess
import threading
import time
import os
import cv2
import numpy as np
import smtplib
from email.message import EmailMessage
import mimetypes
from flask import Flask, Response
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO("yolov8n.pt")

CONF_THRESH = 0.25
PALETTE = {"fire": (0, 0, 255), "smoke": (255, 0, 0)}
FALLBACK_COLOR = (0, 255, 0)
EMAIL_COOLDOWN = 60

last_email_sent = 0.0
current_frame = None
frame_lock = threading.Lock()

def send_email_alert(subject, body, attachment_path=None):
    user = os.getenv("ALERT_EMAIL_USER")
    passwd = os.getenv("ALERT_EMAIL_PASS")
    to_addr = os.getenv("ALERT_EMAIL_TO")
    if not (user and passwd and to_addr):
        print("Email vars not set!")
        return False
    msg = EmailMessage()
    msg["From"] = user
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content(body)
    if attachment_path and os.path.exists(attachment_path):
        ctype, _ = mimetypes.guess_type(attachment_path)
        if ctype is None:
            ctype = "application/octet-stream"
        maintype, subtype = ctype.split("/", 1)
        with open(attachment_path, "rb") as f:
            msg.add_attachment(f.read(), maintype=maintype,
                             subtype=subtype,
                             filename=os.path.basename(attachment_path))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(user, passwd)
            smtp.send_message(msg)
        print("Email alert sent!")
        return True
    except Exception as e:
        print("Failed to send email:", e)
        return False

def capture_and_detect():
    global current_frame, last_email_sent
    print("🎥 Starting camera...")
    process = subprocess.Popen(
        ['rpicam-vid', '-t', '0', '--inline', '-n',
         '--width', '640', '--height', '480',
         '--codec', 'mjpeg', '-o', '-'],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    print("✅ Camera started")
    buf = b''
    frame_idx = 0
    try:
        while True:
            chunk = process.stdout.read(4096)
            if not chunk:
                break
            buf += chunk
            start = buf.find(b'\xff\xd8')
            end = buf.find(b'\xff\xd9')
            if start != -1 and end != -1:
                jpg = buf[start:end+2]
                buf = buf[end+2:]
                frame_idx += 1
                # Decode frame
                frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                # Run detection on every frame
                results = model(frame, verbose=False)
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
                                    (x1, max(20, y1-10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                # Print detections
                if labels_this_frame:
                    pretty = ", ".join([f"{lab}:{conf*100:.1f}%" for lab, conf in labels_this_frame])
                    print(f"🔥 Frame {frame_idx}: {pretty}")
                # Email alert
                now = time.time()
                if labels_this_frame and (now - last_email_sent) > EMAIL_COOLDOWN:
                    snap_path = f"alert_frame_{frame_idx}.jpg"
                    cv2.imwrite(snap_path, frame)
                    top_label, top_conf = max(labels_this_frame, key=lambda x: x[1])
                    subject = f"🔥 ALERT: {top_label.upper()} detected!"
                    body = f"Detected {top_label} with confidence {top_conf*100:.1f}% at frame {frame_idx}."
                    threading.Thread(target=send_email_alert,
                                   args=(subject, body, snap_path), daemon=True).start()
                    last_email_sent = time.time()
                # Store frame for streaming
                ret, buffer = cv2.imencode('.jpg', frame)
                with frame_lock:
                    current_frame = buffer.tobytes()
    except KeyboardInterrupt:
        print("\n⏹️  Stopping camera...")
    finally:
        process.terminate()
        process.wait()
        print("✅ Camera stopped")

def gen_frames():
    while True:
        with frame_lock:
            frame = current_frame
        if frame is None:
            time.sleep(0.1)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.05)

@app.route('/')
def index():
    return Response(gen_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("=" * 60)
    print("🔥 Fire & Smoke Detection System - Raspberry Pi 5")
    print("=" * 60)

    # Check email configuration
    if not all([os.getenv("ALERT_EMAIL_USER"), os.getenv("ALERT_EMAIL_PASS"), os.getenv("ALERT_EMAIL_TO")]):
        print("⚠️  WARNING: Email credentials not set!")
        print("   Set: ALERT_EMAIL_USER, ALERT_EMAIL_PASS, ALERT_EMAIL_TO")
    else:
        print(f"✅ Email alerts enabled: {os.getenv('ALERT_EMAIL_TO')}")

    print(f"📊 Model: yolov8n.pt | Confidence: {CONF_THRESH}")
    print(f"⏱️  Email cooldown: {EMAIL_COOLDOWN}s\n")

    # Start camera thread
    t = threading.Thread(target=capture_and_detect)
    t.daemon = True
    t.start()

    # Wait for first frame
    print("⏳ Waiting for first frame...")
    while current_frame is None:
        time.sleep(0.1)
    print("✅ Stream ready\n")

    print("🌐 Starting web server on http://0.0.0.0:5000")
    print("   Access from browser: http://<raspberry-pi-ip>:5000")
    print("\nPress Ctrl+C to stop\n")

    try:
        app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
    except KeyboardInterrupt:
        print("\n⏹️  Shutting down...")
    print("✅ Done")