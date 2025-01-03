import os
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
import pandas as pd
import math

# Khởi tạo Flask app và YOLO model
app = Flask(__name__)
model = YOLO("D:/f8-shop/cam/best.pt")  # Đường dẫn tới mô hình YOLO

# Thư mục lưu trữ ảnh đầu ra
output_folder = 'D:/f8-shop/cam/images'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Hàm tính tâm bounding box
def calculate_center(bbox):
    x_min, y_min, x_max, y_max = bbox
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    return (center_x, center_y)

# Hàm tính kích thước bounding box (mm)
def calculate_bbox_size(bbox, pixel_to_mm_ratio=1.31):
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min
    width_mm = width * pixel_to_mm_ratio
    height_mm = height * pixel_to_mm_ratio
    return width_mm, height_mm

# Hàm tính diện tích và khoảng cách
def calculate_additional_metrics(center, bbox, height, width, pixel_to_mm_ratio=1.31):
    # Tính diện tích của bounding box (mm²)
    width_mm, height_mm = calculate_bbox_size(bbox, pixel_to_mm_ratio)
    area_mm2 = width_mm * height_mm

    # Tính tọa độ giao điểm của đường vuông góc từ tâm bbox xuống cạnh dưới ảnh
    intersection_point = (center[0], height-1)

    # Tính khoảng cách từ tâm bounding box đến intersection_point
    a_pixel = math.sqrt((intersection_point[0] - center[0]) ** 2 + (intersection_point[1] - center[1]) ** 2)
    a_mm = a_pixel * pixel_to_mm_ratio

    # Tính tọa độ chính giữa cạnh dưới của ảnh
    center_bottom = (width // 2, height - 1)

    # Tính khoảng cách từ intersection_point đến center_bottom
    d1_pixel = math.sqrt((center_bottom[0] - intersection_point[0]) ** 2 + (center_bottom[1] - intersection_point[1]) ** 2)
    d1_mm = d1_pixel * pixel_to_mm_ratio
    
    # Tính giá trị x và y từ hệ phương trình
    y = d1_mm / (1 + (a_mm / 550))
    x = (a_mm / 550) * y
    
    # Khoảng cách từ camera đến FOD
    d_end = math.sqrt((a_mm) ** 2 + (x) ** 2) + math.sqrt((y) ** 2 + 550 ** 2)  # Kích thước đo từ camera đến vật thể
    
    return area_mm2, width_mm, height_mm, d_end

# Video capture
def gen_frames():
    cap = cv2.VideoCapture(1)  # Mở webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Phát hiện đối tượng và lấy kết quả
        results = model(frame)
        image_resized = results.render()[0]  # Vẽ bounding boxes lên ảnh
        objects = []
        stats = []
        height, width, _ = image_resized.shape  # Kích thước ảnh đã thay đổi

        for result in results[0].boxes.xyxy:  # Lấy các bounding box
            x_min, y_min, x_max, y_max = map(int, result.tolist())
            center = calculate_center((x_min, y_min, x_max, y_max))
            
            # Tính toán các chỉ số
            area_mm2, width_mm, height_mm, d_end = calculate_additional_metrics(center, (x_min, y_min, x_max, y_max), height, width)
            
            # Thêm kết quả vào danh sách
            objects.append({
                "bbox": (x_min, y_min, x_max, y_max),
                "name": model.names[int(results[0].boxes.cls[0])],
                "area_mm2": area_mm2,
                "width_mm": width_mm,
                "height_mm": height_mm,
                "distance_to_FOD_mm": d_end
            })

            # Cập nhật bảng thống kê
            stats.append({
                'Object Name': model.names[int(results[0].boxes.cls[0])],
                'Area (mm^2)': area_mm2,
                'Width (mm)': width_mm,
                'Height (mm)': height_mm,
                'Distance to FOD (mm)': d_end
            })

        # Lưu kết quả ảnh đã xử lý và bảng CSV
        image_path = os.path.join(output_folder, "result.jpg")
        cv2.imwrite(image_path, image_resized)
        
        # Chuyển đổi ảnh sang định dạng JPEG cho trình duyệt
        _, buffer = cv2.imencode('.jpg', image_resized)
        frame_data = buffer.tobytes()

        # Lưu CSV
        df = pd.DataFrame(stats)
        csv_file = os.path.join(output_folder, "stats.csv")
        df.to_csv(csv_file, index=False)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n\r\n')

# Định nghĩa trang chủ
@app.route('/')
def index():
    return render_template('index.html')

# Định nghĩa route cho video streaming
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route để tải dữ liệu CSV (nếu cần)
@app.route('/get_csv')
def get_csv():
    csv_path = os.path.join(output_folder, 'stats.csv')
    with open(csv_path, 'r') as file:
        data = file.read()
    return jsonify({'csv_data': data})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
