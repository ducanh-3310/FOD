from ultralytics import YOLO
import cv2
import numpy as np
import pickle
import math
import pandas as pd
import os
import glob
import yaml

# Đọc tệp data.yaml để lấy danh sách các lớp vật thể
def load_class_names(yaml_file):
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)
    return data['names']  # Trả về danh sách tên các vật thể từ 'names' trong file YAML

# Đường dẫn đến tệp data.yaml
yaml_file_path = '/home/ducanh/Downloads/results/data.yaml'  # Cập nhật đường dẫn đến tệp data.yaml của bạn
class_names = load_class_names(yaml_file_path)  # Tải tên các lớp từ file data.yaml


# Các hàm hỗ trợ
def calculate_center(bbox):
    x_min, y_min, x_max, y_max = bbox
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    return (center_x, center_y)

def calculate_bbox_size(bbox, pixel_to_mm_ratio=1.31):
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min
    width_mm = width * pixel_to_mm_ratio
    height_mm = height * pixel_to_mm_ratio
    return width_mm, height_mm

def transform_perspective(points, perspective_matrix):
    points_homogeneous = np.hstack([points, np.ones((len(points), 1))])
    transformed_points = points_homogeneous @ perspective_matrix.T
    transformed_points = transformed_points[:, :2] / transformed_points[:, 2, np.newaxis]
    return transformed_points

def undistort_points(points, camera_matrix, dist_coeffs):
    points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    undistorted = cv2.undistortPoints(points, camera_matrix, dist_coeffs, P=camera_matrix)
    return undistorted.reshape(-1, 2)

def load_matrix(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

# Tải các ma trận từ file
perspective_matrix = load_matrix("/home/ducanh/FOD/cam0/perspective_matrix.pkl")
camera_matrix = load_matrix("/home/ducanh/FOD/cam0/cameraMatrix.pkl")
dist_coeffs = load_matrix("/home/ducanh/FOD/cam0/dist.pkl")

# Tải mô hình YOLO từ file best.pt
model = YOLO("/home/ducanh/Downloads/results/runs/detect/train/weights/best.pt")  # Đường dẫn tới file best.pt

# Đọc các tệp ảnh từ thư mục đầu vào
input_folder = "/home/ducanh/FOD/cam0/check/"  # Đường dẫn tới thư mục chứa các ảnh đầu vào
output_folder = "/home/ducanh/FOD/cam0/imgend/"  # Đường dẫn tới thư mục lưu ảnh đã xử lý

# Kiểm tra và tạo thư mục output nếu không tồn tại
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Lấy danh sách các tệp ảnh từ thư mục đầu vào
image_paths = glob.glob(os.path.join(input_folder, "*.jpeg"))  # Bạn có thể thay đổi theo định dạng ảnh (jpeg, png, v.v.)

# Tạo danh sách để lưu kết quả thống kê của tất cả các ảnh
all_stats = []

# Lặp qua từng ảnh trong thư mục
for image_path in image_paths:
    # Đọc ảnh đầu vào
    image = cv2.imread(image_path)
    
    # Thay đổi kích thước ảnh đầu vào thành 800x450
    image_resized = cv2.resize(image, (800, 450))
    
    # Phát hiện đối tượng qua YOLO
    results = model(image_resized)

    # Lấy bounding boxes và tên của các vật thể từ YOLO
    bounding_boxes = []
    object_names = []
    for result in results[0].boxes.xyxy:  # Dạng [x_min, y_min, x_max, y_max]
        x_min, y_min, x_max, y_max = map(int, result.tolist())
        bounding_boxes.append((x_min, y_min, x_max, y_max))
    # for label in results[0].names:  # Lấy tên các vật thể đã phát hiện
    #     object_names.append(label)
    # Lấy tên các vật thể đã phát hiện
    # object_names = [model.names[int(label)] for label in results[0].boxes.cls]
    object_names = [class_names[int(label)] for label in results[0].boxes.cls]


    # Nếu không phát hiện vật thể nào, bỏ qua các phép toán tiếp theo
    if not bounding_boxes:
        print(f"No objects detected in image: {image_path}")
        continue  # Tiến hành với ảnh tiếp theo nếu không phát hiện vật thể nào

    # Tính tâm của bounding boxes
    centers = [calculate_center(bbox) for bbox in bounding_boxes]

    # Chuyển đổi tọa độ qua ma trận phối cảnh
    transformed_centers = transform_perspective(np.array(centers), perspective_matrix)

    # Hiệu chỉnh méo ảnh
    undistorted_centers = undistort_points(transformed_centers, camera_matrix, dist_coeffs)

    # Lấy chiều cao và chiều rộng của ảnh đã thay đổi kích thước
    height, width, _ = image_resized.shape

    # Tạo danh sách để lưu kết quả thống kê của ảnh hiện tại
    stats = []

    # Hiển thị bounding boxes, các tâm, và in kích thước thực tế
    for idx, (bbox, center, object_name) in enumerate(zip(bounding_boxes, centers, object_names), start=1):
        x_min, y_min, x_max, y_max = bbox
        width_mm, height_mm = calculate_bbox_size(bbox, pixel_to_mm_ratio=1.31)
        
        # Tính diện tích của bounding box (mm²)
        area_mm2 = width_mm * height_mm
        
        # Tính tọa độ giao điểm của đường vuông góc từ tâm bbox xuống cạnh dưới ảnh
        intersection_point = (center[0], height-1)
        
        # Tính khoảng cách từ tâm bounding box đến intersection_point
        a_pixel = math.sqrt((intersection_point[0] - center[0]) ** 2 + (intersection_point[1] - center[1]) ** 2)
        a_mm = a_pixel * 1.31

        # Tính tọa độ chính giữa cạnh dưới của ảnh
        center_bottom = (width // 2, height - 1)

        # Tính khoảng cách từ intersection_point đến center_bottom
        d1_pixel = math.sqrt((center_bottom[0] - intersection_point[0]) ** 2 + (center_bottom[1] - intersection_point[1]) ** 2)
        d1_mm = d1_pixel * 1.31
        
        # Tính giá trị x và y từ hệ phương trình
        y = d1_mm / (1 + (a_mm / 550))
        x = (a_mm / 550) * y
        
        # Khoảng cách từ camera đến FOD
        d_end = math.sqrt((a_mm) ** 2 + (x) ** 2) + math.sqrt((y) ** 2 + 550 ** 2)  # Kích thước đo từ camera đến vật thể
        
        # Thêm thông tin vào danh sách thống kê
        stats.append({
            'Object No': idx,
            'Object Name': object_name,  # Thêm tên vật thể
            'Area (mm^2)': area_mm2,
            'Height (mm)':height_mm,
            'Width (mm)': width_mm,
            'Distance to FOD (mm)': d_end
        })
        
        # Vẽ bounding box và số thứ tự
        cv2.rectangle(image_resized, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.circle(image_resized, center, 5, (0, 0, 255), -1)
        cv2.putText(image_resized, f"{object_name} {idx}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Chuyển danh sách thống kê thành DataFrame và lưu ra file CSV cho mỗi ảnh
    df = pd.DataFrame(stats)
    stats_csv_path = os.path.join(output_folder, os.path.basename(image_path).replace(".jpeg", "_stats.csv"))
    df.to_csv(stats_csv_path, index=False)

    # Lưu ảnh đã xử lý vào thư mục output
    output_image_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_image_path, image_resized)

    print(f"Processed and saved image: {output_image_path}")

    # Thêm kết quả của ảnh vào danh sách tổng thể
    all_stats.extend(stats)

# Sau khi xử lý tất cả các ảnh, lưu kết quả tổng thể vào file CSV
all_stats_df = pd.DataFrame(all_stats)
all_stats_csv_path = os.path.join(output_folder, "all_stats.csv")
all_stats_df.to_csv(all_stats_csv_path, index=False)

print("Processing completed for all images.")
