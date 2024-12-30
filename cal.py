from ultralytics import YOLO
import cv2
import numpy as np
import pickle
import math

# Tải mô hình YOLO từ file best.pt
model = YOLO("/home/ducanh/Downloads/results/runs/detect/train/weights/best.pt")  # Đường dẫn tới file best.pt

# Đọc ảnh đầu vào
image_path = "/home/ducanh/FOD/cam0/test/img9.jpeg"  # Đường dẫn tới ảnh
image = cv2.imread(image_path)

# Thay đổi kích thước ảnh đầu vào thành 800x450
image_resized = cv2.resize(image, (800, 450))

# Phát hiện đối tượng qua YOLO
results = model(image_resized)

# Lấy bounding boxes từ YOLO
bounding_boxes = []
for result in results[0].boxes.xyxy:  # Dạng [x_min, y_min, x_max, y_max]
    x_min, y_min, x_max, y_max = map(int, result.tolist())
    bounding_boxes.append((x_min, y_min, x_max, y_max))

# Hàm tính tâm của bounding box
def calculate_center(bbox):
    x_min, y_min, x_max, y_max = bbox
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    return (center_x, center_y)

# Hàm tính kích thước thực tế của bounding box (dùng đơn vị mm)
def calculate_bbox_size(bbox, pixel_to_mm_ratio=1.31):
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min
    
    # Chuyển từ pixel sang mm
    width_mm = width * pixel_to_mm_ratio
    height_mm = height * pixel_to_mm_ratio
    
    return width_mm, height_mm

# Tính tâm của bounding boxes
centers = [calculate_center(bbox) for bbox in bounding_boxes]

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

# Chuyển đổi tọa độ qua ma trận phối cảnh
transformed_centers = transform_perspective(np.array(centers), perspective_matrix)

# Hiệu chỉnh méo ảnh
undistorted_centers = undistort_points(transformed_centers, camera_matrix, dist_coeffs)

# Lấy chiều cao và chiều rộng của ảnh đã thay đổi kích thước
height, width, _ = image_resized.shape

# Hiển thị bounding boxes và các tâm, đồng thời in kích thước thực tế
for bbox, center in zip(bounding_boxes, centers):
    x_min, y_min, x_max, y_max = bbox
    width_mm, height_mm = calculate_bbox_size(bbox, pixel_to_mm_ratio=1.31)
    
    # In kích thước bounding box thực tế (mm)
    print(f"Bounding box (pixel): {bbox}")
    print(f"Bounding box (mm): Width = {width_mm:.2f} mm, Height = {height_mm:.2f} mm")
    
    # In tọa độ của tâm bounding box theo pixel
    print(f"Center (pixel): {center}")
    
    # Tính tọa độ giao điểm của đường vuông góc từ tâm bbox xuống cạnh dưới ảnh
    intersection_point = (center[0], height-1)
    print(f"Intersection point (pixel): {intersection_point}")

    # Tính tọa độ chính giữa cạnh dưới của ảnh
    center_bottom = (width // 2, height - 1)
    print(f"Center bottom (pixel): {center_bottom}")

    # Tính khoảng cách từ intersection_point đến center_bottom
    d1_pixel = math.sqrt((center_bottom[0] - intersection_point[0]) ** 2 + (center_bottom[1] - intersection_point[1]) ** 2)
    
    # Chuyển khoảng cách từ pixel sang mm
    d1_mm = d1_pixel * 1.31
    print(f"Distance from intersection point to center bottom: {d1_mm:.2f} mm")
    
    # Vẽ bounding box và các tâm
    cv2.rectangle(image_resized, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.circle(image_resized, center, 5, (0, 0, 255), -1)

    # Vẽ đường thẳng vuông góc từ tâm bounding box xuống cạnh dưới của khung hình (không phải bbox)
    cv2.line(image_resized, center, (center[0], height-1), (255, 0, 0), 2)
    
    # Tính chiều dài của cạnh vuông góc (từ tâm bounding box đến cạnh dưới của khung hình)
    distance_pixels = height - center[1]
    
    # Chuyển từ pixel sang mm
    distance_mm = distance_pixels * 1.31
    print(f"Distance from center to bottom of image: {distance_mm:.2f} mm")
    
    # Khoảng cách từ camera đến FOD
    d_end = math.sqrt((d1_mm / 2) ** 2 + (distance_mm) ** 2) + math.sqrt((d1_mm / 2) ** 2 + 600 ** 2) #kcach tu chan cam den chinh giua canh giua anh la 60cm
    print(f"Distance from camera to FOD: {d_end:.2f} mm")

# Hiển thị ảnh kết quả có kích thước 800x450
cv2.imshow("Result", image_resized)
output_image_path = "/home/ducanh/FOD/cam0/done_1.jpg"
cv2.imwrite(output_image_path, image_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()




# yolo detect predict model=yolo11n.pt source='https://ultralytics.com/images/bus.jpg'  # predict with official model
# yolo detect predict model=/home/ducanh/Downloads/results/runs/detect/train/weights/best.pt source='/home/ducanh/FOD/cam0/Pasted image (2).png'  # predict with custom model