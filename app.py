import cv2
from flask import Flask, Response, render_template

app = Flask(__name__)

# Khởi tạo webcam (0 là chỉ định webcam mặc định)
video_capture = cv2.VideoCapture(1)

def generate_frames():
    while True:
        # Đọc từng frame từ webcam
        ret, frame = video_capture.read()

        if not ret:
            break

        # Chuyển đổi frame từ BGR sang JPEG
        ret, buffer = cv2.imencode('.jpg', frame)

        if not ret:
            continue

        # Chuyển frame thành định dạng byte và gửi qua HTTP
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')  # Trả về file HTML

@app.route('/video_feed')
def video_feed():
    # Trả về video stream
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
