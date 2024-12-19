############################ # dua anh tu goc nghieng ve goc thang voi 4 toa do cho truoc
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc camera matrix và distortion coefficients đã lưu
with open("/home/ducanh/FOD/cam0/cameraMatrix.pkl", "rb") as f:
    camera_matrix_data = pickle.load(f)

with open("/home/ducanh/FOD/cam0/dist.pkl", "rb") as f:
    distortion_data = pickle.load(f)

# Trích xuất camera matrix và distortion coefficients
mtx = camera_matrix_data
dist = distortion_data[0]
print("Shape of Camera Matrix (mtx):", mtx.shape)
print("Shape of Distortion Coefficients (dist):", dist.shape)

# Đọc ảnh đầu vào và thay đổi kích thước về 800x450
img = cv2.imread('/home/ducanh/FOD/cam0/images/img0.jpeg')
img = cv2.resize(img, (800, 450))  # Đổi kích thước ảnh về 800x450

img_size = (img.shape[1], img.shape[0])  # Kích thước của ảnh sau khi resize

nx = 11  # Số góc trong chiều x của bàn cờ
ny = 17  # Số góc trong chiều y của bàn cờ

# Hàm chuyển đổi phối cảnh (undistort và perspective transform)
def corners_unwarp(img, nx, ny, mtx, dist):
    # Đảo ảnh theo chiều dọc trước khi xử lý
    # img = cv2.flip(img, 0)  # 0 là lật ảnh theo chiều dọc

    # Undistort ảnh bằng camera matrix và distortion coefficients
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    
    # Chuyển sang ảnh xám
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    
    # Tìm các góc của bàn cờ
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    
    warped = None
    M = None
    
    # Nếu tìm thấy các góc
    if ret == True:
        # Vẽ các góc lên ảnh
        cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
        
        # Định nghĩa 4 điểm nguồn (src) từ các góc đã phát hiện
        src = np.float32([corners[0], corners[nx - 1], corners[-1], corners[-nx]])

        # Định nghĩa 4 điểm đích (dst) cho phép chuyển phối cảnh
        dst = np.float32([[576, 217], [576, 367], [336, 367], [336, 217]])
        
        # Tính ma trận chuyển đổi phối cảnh M
        M = cv2.getPerspectiveTransform(src, dst)
        
        # Dùng ma trận M để biến đổi ảnh vào góc nhìn từ trên xuống
        warped = cv2.warpPerspective(undist, M, img_size, flags=cv2.INTER_LINEAR)
        
        # Lưu ma trận chuyển đổi phối cảnh vào file pickle
        with open('/home/ducanh/FOD/cam0/perspective_matrix.pkl', 'wb') as f:
            pickle.dump(M, f)
        print("Đã lưu ma trận chuyển đổi phối cảnh vào file 'perspective_matrix.pkl'.")
    
    return warped, M

# Gọi hàm để thực hiện undistort và warp ảnh
warped, perspective_M = corners_unwarp(img, nx, ny, mtx, dist)

# Hiển thị ảnh gốc và ảnh đã biến đổi (undistorted và warped)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()

# Chuyển ảnh BGR sang RGB để hiển thị đúng màu
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  
ax1.set_title('Original Image', fontsize=50)

# Chuyển ảnh BGR sang RGB để hiển thị đúng màu
ax2.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))  
ax2.set_title('Undistorted and Warped Image', fontsize=50)

plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

# Lưu kết quả vào file ảnh
plt.savefig('output_0.jpeg')

