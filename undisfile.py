import pickle
import cv2
import numpy as np
import os

# Đọc ma trận camera, hệ số sai lệch và ma trận chuyển đổi phối cảnh đã lưu
with open("/home/ducanh/FOD/cam0/cameraMatrix.pkl", "rb") as f:
    camera_matrix_data = pickle.load(f)

with open("/home/ducanh/FOD/cam0/dist.pkl", "rb") as f:
    distortion_data = pickle.load(f)

with open('/home/ducanh/FOD/cam0/perspective_matrix.pkl', 'rb') as f:
    perspective_matrix = pickle.load(f)

# Trích xuất camera matrix và distortion coefficients
mtx = camera_matrix_data
dist = distortion_data[0]
M = perspective_matrix

# Đọc thư mục chứa ảnh đầu vào
input_folder = '/home/ducanh/FOD/cam0/test/'  # Thay đổi đường dẫn tới thư mục ảnh của bạn
output_folder = '/home/ducanh/FOD/cam0/check/'  # Thư mục lưu ảnh đã xử lý

# Tạo thư mục lưu ảnh đầu ra nếu chưa có
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Lặp qua tất cả các tệp ảnh trong thư mục đầu vào
image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpeg') or f.endswith('.jpg')]

for idx, image_file in enumerate(image_files):
    # Đọc ảnh đầu vào
    img = cv2.imread(os.path.join(input_folder, image_file))

    if img is None:
        print(f"Không thể đọc ảnh {image_file}")
        continue

    # Thay đổi kích thước ảnh về 800x450
    img_resized = cv2.resize(img, (800, 450))

    # Sửa sai lệch ảnh sử dụng ma trận camera và hệ số sai lệch
    undistorted_img = cv2.undistort(img_resized, mtx, dist, None, mtx)

    # Lấy kích thước của ảnh
    img_size = (undistorted_img.shape[1], undistorted_img.shape[0])

    # Áp dụng chuyển đổi phối cảnh lên ảnh đã sửa sai lệch
    warped_img = cv2.warpPerspective(undistorted_img, M, img_size, flags=cv2.INTER_LINEAR)

    # Tạo tên file đầu ra theo định dạng 'check+number'
    output_filename = f'check{idx+1}'

    # Lưu ảnh đã chuyển đổi phối cảnh vào thư mục đầu ra
    cv2.imwrite(os.path.join(output_folder, output_filename + '.jpeg'), warped_img)

    print(f"Đã xử lý và lưu ảnh: {output_filename}.jpeg")

print("Xử lý tất cả ảnh hoàn tất.")
