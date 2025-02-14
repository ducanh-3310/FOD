from ultralytics import YOLO
import cv2
import numpy as np
import pickle
import math
import yaml
import os

log_path= "/home/ducanh/FOD/cam0/log/log.log"
# Tải mô hình YOLO từ file best.pt
model = YOLO("/home/ducanh/FOD/best.pt")  # Đường dẫn tới file best.pt

# Mở video đầu vào
video_path = "/home/ducanh/FOD/cam0/video/cam0.webm"  # Đường dẫn tới video
cap = cv2.VideoCapture(video_path)

# Kiểm tra nếu video mở thành công
if not cap.isOpened():
    print("Không thể mở video.")
    exit()

# Lấy thông tin về video
fps = cap.get(cv2.CAP_PROP_FPS)  # Tốc độ khung hình (FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Mở file để lưu video đầu ra
output_path = "/home/ducanh/FOD/cam0/vdout/video_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Hàm tính tâm của bounding box
def calculate_center(bbox):
    x_min, y_min, x_max, y_max = bbox
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    return (center_x, center_y)

# Hàm tính kích thước thực tế của bounding box (dùng đơn vị mm)
def calculate_bbox_size(bbox, pixel_to_mm_ratio=0.131):
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min
    
    # Chuyển từ pixel sang mm
    width_mm = width * pixel_to_mm_ratio
    height_mm = height * pixel_to_mm_ratio
    
    if width_mm > height_mm:
        width_mm, height_mm = height_mm, width_mm

    return width_mm, height_mm

# Hàm chuyển đổi qua ma trận phối cảnh
def transform_perspective(points, perspective_matrix):
    points_homogeneous = np.hstack([points, np.ones((len(points), 1))])
    transformed_points = points_homogeneous @ perspective_matrix.T
    transformed_points = transformed_points[:, :2] / transformed_points[:, 2, np.newaxis]
    return transformed_points

# Hàm hiệu chỉnh méo ảnh
def undistort_points(points, camera_matrix, dist_coeffs):
    points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    undistorted = cv2.undistortPoints(points, camera_matrix, dist_coeffs, P=camera_matrix)
    return undistorted.reshape(-1, 2)

# Tải các ma trận từ file
def load_matrix(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

perspective_matrix = load_matrix("/home/ducanh/FOD/cam0/perspective_matrix.pkl")
camera_matrix = load_matrix("/home/ducanh/FOD/cam0/cameraMatrix.pkl")
dist_coeffs = load_matrix("/home/ducanh/FOD/cam0/dist.pkl")

def load_class_names(yaml_path):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return data['names']

# Đường dẫn tới file data.yaml
yaml_file = '/home/ducanh/FOD/data.yaml'  # anh xa toi file da.yaml de doc ten fod
class_names = load_class_names(yaml_file)

# Mở file log với chế độ append (a) để thêm thông tin vào cuối file thay vì ghi đè
with open(log_path, mode='a') as log_file:
    i=0
    # Lặp qua từng khung hình của video
    while cap.isOpened():
        ret, frame = cap.read()    
        if not ret:
            break
        
        # Thay đổi kích thước ảnh đầu vào thành 800x450
        image_resized = cv2.resize(frame, (800, 450))
        
        # Phát hiện đối tượng qua YOLO
        results = model(image_resized)
        #results = model(image_resized, verbose=False)   #tat hien thi thong bao tu yolo


        # Lấy bounding boxes từ YOLO
        bounding_boxes = []
        for result in results[0].boxes.xyxy:  # Dạng [x_min, y_min, x_max, y_max]
            x_min, y_min, x_max, y_max = map(int, result.tolist())
            bounding_boxes.append((x_min, y_min, x_max, y_max))

        if len(bounding_boxes) == 0:
                continue  # Nếu không có bounding box, bỏ qua frame này
        
        labels = []
        for result in results[0].boxes.cls:  # Lấy ID lớp (class IDs)
            label_id = int(result.item())  # Convert Tensor to integer
            labels.append(class_names[label_id])  # Ánh xạ từ ID lớp sang tên lớp

        # Tính tâm của bounding boxes
        centers = [calculate_center(bbox) for bbox in bounding_boxes]

        # Chuyển đổi tọa độ qua ma trận phối cảnh
        transformed_centers = transform_perspective(np.array(centers), perspective_matrix)

        # Hiệu chỉnh méo ảnh
        undistorted_centers = undistort_points(transformed_centers, camera_matrix, dist_coeffs)

        # Lấy chiều cao và chiều rộng của ảnh đã thay đổi kích thước
        height, width, _ = image_resized.shape

        i+=1
        print("frame %d " % i)
        log_file.write(f"Frame {i}:\n")
        z=0
        # Hiển thị bounding boxes và các tâm, đồng thời in kích thước thực tế
        for bbox, center, label in zip(bounding_boxes, centers, labels):
            z+=1
            x_min, y_min, x_max, y_max = bbox
            width_mm, height_mm = calculate_bbox_size(bbox, pixel_to_mm_ratio=0.131)
            
            
            # In kích thước bounding box thực tế (mm)
            print(f"No.{z} ({label}): Width = {width_mm:.1f} cm, Height = {height_mm:.1f} cm")
            # In số thứ tự và tên vật thể cùng với kích thước vào chuỗi
            s = f"No.{z} ({label}): Width = {width_mm:.1f} cm, Height = {height_mm:.1f} cm\n"

            # Ghi thông tin vào file log
            log_file.write(s)

            # Tính tọa độ giao điểm của đường vuông góc từ tâm bbox xuống cạnh dưới ảnh
            intersection_point = (center[0], height-1)

            # Tính khoảng cách từ tâm bounding box đến intersection_point
            a_pixel = math.sqrt((intersection_point[0] - center[0]) ** 2 + (intersection_point[1] - center[1]) ** 2)
            a_mm = a_pixel * 0.131

            # Tính tọa độ chính giữa cạnh dưới của ảnh
            center_bottom = (width // 2, height - 1)

            # Tính khoảng cách từ intersection_point đến center_bottom
            d1_pixel = math.sqrt((center_bottom[0] - intersection_point[0]) ** 2 + (center_bottom[1] - intersection_point[1]) ** 2)
            d1_mm = d1_pixel * 0.131
            
            # Tính giá trị x và y từ hệ phương trình
            y = d1_mm / (1 + (a_mm / 55))
            x = (a_mm / 55) * y

            # Vẽ bounding box và các tâm
            cv2.rectangle(image_resized, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.circle(image_resized, center, 5, (0, 0, 255), -1)

            # Vẽ đường thẳng vuông góc từ tâm bounding box xuống cạnh dưới của khung hình (không phải bbox)
            cv2.line(image_resized, center, (center[0], height-1), (255, 0, 0), 2)

            # Khoảng cách từ camera đến FOD
            d_end = math.sqrt((a_mm) ** 2 + (x) ** 2) + math.sqrt((y) ** 2 + 55 ** 2) #kcach tu chan cam den chinh giua canh giua anh la 55cm
            print(f"Distance from camera to object No.{z}: {d_end:.1f} cm")
            log_file.write(f"Distance from camera to object No.{z}: {d_end:.1f} cm\n")


            # In tên vật thể lên ảnh
            cv2.putText(image_resized, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        # Ghi khung hình đã xử lý vào video đầu ra
        out.write(image_resized)

        # Hiển thị video trong thời gian thực
        cv2.imshow("Result", image_resized)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Giải phóng tài nguyên
cap.release()
out.release()
cv2.destroyAllWindows()
