import subprocess
from flask import Flask, Response

app = Flask(__name__)

def gen_frames():
    process = subprocess.Popen(
        ['rpicam-vid', '-t', '0', '--inline', '-n',
         '--width', '640', '--height', '480',
         '--codec', 'mjpeg', '-o', '-'],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    
    buf = b''
    while True:
        buf += process.stdout.read(4096)
        start = buf.find(b'\xff\xd8')
        end = buf.find(b'\xff\xd9')
        if start != -1 and end != -1:
            frame = buf[start:end+2]
            buf = buf[end+2:]
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return Response(gen_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)