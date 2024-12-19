####################phat hien checkerboard va tinh toan ti le mm/pixel, su dung anh sau khi thuc hien tat ca bien doi
import cv2
import numpy as np

# Định nghĩa kích thước của checkerboard (16x10)
checkerboard_size = (17, 11)  # (Số ô vuông theo chiều ngang, chiều dọc)

# Kích thước mỗi ô vuông (20mm x 20mm)
square_size_mm = 20  # mm

# Đọc ảnh từ file (thay vì từ webcam)
img = cv2.imread('/home/ducanh/FOD/cam0/check/check10.jpeg')  # Thay đổi đường dẫn tới ảnh của bạn

# Kiểm tra xem ảnh có được đọc thành công không
if img is None:
    print("Không thể đọc hình ảnh từ file")
else:
    # Chuyển ảnh sang chế độ xám
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Tìm các góc checkerboard
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

    if ret:  # Nếu tìm thấy các góc của checkerboard
        # Vẽ các điểm góc lên hình ảnh
        cv2.drawChessboardCorners(img, checkerboard_size, corners, ret)

        # Tính toán bounding box xung quanh checkerboard
        # Tìm min và max của các tọa độ góc để tạo bounding box
        x_min = np.min(corners[:, 0, 0])
        x_max = np.max(corners[:, 0, 0])
        y_min = np.min(corners[:, 0, 1])
        y_max = np.max(corners[:, 0, 1])

        # Vẽ bounding box
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

        # Tính toán tỷ lệ pixel/mm
        width_pixels = x_max - x_min
        height_pixels = y_max - y_min

        # Tính kích thước thực tế
        width_mm = (checkerboard_size[0] - 1) * square_size_mm  # (16 - 1) * 20 mm = 300 mm
        height_mm = (checkerboard_size[1] - 1) * square_size_mm  # (10 - 1) * 20 mm = 180 mm

        # Tính tỷ lệ pixel/mm
        pixel_per_mm_width = width_pixels / width_mm
        pixel_per_mm_height = height_pixels / height_mm

        # Tính tỷ lệ mm/pixel, đảo ngược tỷ lệ pixel/mm
        mm_per_pixel_width = 1 / pixel_per_mm_width
        mm_per_pixel_height = 1 / pixel_per_mm_height

        # Tính tỷ lệ trung bình mm/pixel
        mm_per_pixel_avg = (mm_per_pixel_width + mm_per_pixel_height) / 2

        print(f"Tỷ lệ mm/pixel theo chiều rộng: {mm_per_pixel_width:.2f}")
        print(f"Tỷ lệ mm/pixel theo chiều cao: {mm_per_pixel_height:.2f}")
        print(f"Tỷ lệ mm/pixel trung bình: {mm_per_pixel_avg:.2f}")

        # Hiển thị tỷ lệ mm trên pixel trên ảnh
        cv2.putText(img, f"Ratio: {mm_per_pixel_avg:.2f} mm/pixel", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Hiển thị ảnh kết quả
        cv2.imshow("Checkerboard Detection", img)

        # Đợi cho đến khi nhấn phím bất kỳ
        cv2.waitKey(0)  # Đợi cho đến khi nhấn phím bất kỳ
        cv2.destroyAllWindows()

    else:
        print("Không tìm thấy checkerboard trong ảnh.")


# # # chuyen anh tu anh ban dau ve 800*450 va xu li anh nho cac ma tran
# import pickle
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Đọc ma trận camera, hệ số sai lệch và ma trận chuyển đổi phối cảnh đã lưu
# with open("/home/ducanh/FOD/cam0/cameraMatrix.pkl", "rb") as f:
#     camera_matrix_data = pickle.load(f)

# with open("/home/ducanh/FOD/cam0/dist.pkl", "rb") as f:
#     distortion_data = pickle.load(f)

# with open('/home/ducanh/FOD/cam0/perspective_matrix.pkl', 'rb') as f:
#     perspective_matrix = pickle.load(f)

# # Trích xuất camera matrix và distortion coefficients
# mtx = camera_matrix_data
# dist = distortion_data[0]
# M = perspective_matrix

# # Đọc ảnh đầu vào
# img = cv2.imread('/home/ducanh/FOD/cam0/test/check0.jpeg')

# # Thay đổi kích thước ảnh về 800x450
# img_resized = cv2.resize(img, (800, 450))

# # Sửa sai lệch ảnh sử dụng ma trận camera và hệ số sai lệch
# undistorted_img = cv2.undistort(img_resized, mtx, dist, None, mtx)

# # Lấy kích thước của ảnh
# img_size = (undistorted_img.shape[1], undistorted_img.shape[0])

# # Áp dụng chuyển đổi phối cảnh lên ảnh đã sửa sai lệch
# warped_img = cv2.warpPerspective(undistorted_img, M, img_size, flags=cv2.INTER_LINEAR)

# # Hiển thị ảnh gốc, ảnh đã sửa sai lệch và ảnh đã chuyển đổi phối cảnh
# f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
# f.tight_layout()

# # # Chuyển ảnh BGR sang RGB để hiển thị đúng màu
# # ax1.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))  
# # ax1.set_title('Resized Image (800x450)', fontsize=30)

# # ax2.imshow(cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB))  
# # ax2.set_title('Undistorted Image', fontsize=30)

# # ax3.imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB))  
# # ax3.set_title('Undistorted and Warped Image', fontsize=30)

# # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
# # plt.savefig('output_transformed_image0.jpeg')

# # Lưu ảnh cuối cùng (ảnh đã chuyển đổi phối cảnh) vào một file riêng biệt
# # cv2.imwrite('check0.jpeg', warped_img)  # Lưu ảnh cuối cùng





