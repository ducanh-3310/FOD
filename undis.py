# dua anh bi meo ve anh binh thuong qua 2 file pkl

import cv2
import numpy as np
import pickle

# Load camera matrix and distortion coefficients
with open("/home/ducanh/FOD/cam0/cameraMatrix.pkl", "rb") as f:
    camera_matrix_data = pickle.load(f)

with open("/home/ducanh/FOD/cam0/dist.pkl", "rb") as f:
    distortion_data = pickle.load(f)

mtx = camera_matrix_data
dist = distortion_data[0]

# Đọc ảnh gốc
img = cv2.imread('/home/ducanh/FOD/cam0/images/img0.jpeg')

# Thay đổi kích thước ảnh gốc về 800x450
img_resized = cv2.resize(img, (800, 450))  # Thay đổi kích thước ảnh

# Giải biến dạng ảnh đã thay đổi kích thước
undist = cv2.undistort(img_resized, mtx, dist, None, mtx)

# Đảm bảo rằng ảnh sau khi giải biến dạng có kích thước 800x450
undist_resized = cv2.resize(undist, (800, 450))

gray = cv2.cvtColor(undist_resized, cv2.COLOR_BGR2GRAY)
# Lưu ảnh đã giải biến dạng và có kích thước 800x450
cv2.imwrite("undis_0.jpeg", undist_resized)

# Nếu cần hiển thị ảnh
# cv2.imshow("Undistorted and Resized Image", undist_resized)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
