import cv2
import numpy as np

# Hàm để xử lý sự kiện click chuột và lưu tọa độ
def click_event(event, x, y, flags, param):
    global points  # Danh sách lưu các điểm click chuột
    if event == cv2.EVENT_LBUTTONDOWN:
        # Lưu tọa độ của điểm được click vào
        points.append((x, y))
        # Vẽ điểm tại vị trí đã click
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Image", img)
        
        # Nếu đã click 2 điểm, tính khoảng cách giữa chúng
        if len(points) == 2:
            # Tính khoảng cách giữa hai điểm (sử dụng định lý Pythagoras)
            x1, y1 = points[0]
            x2, y2 = points[1]
            pixel_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Tính chiều dài thực tế
            real_distance = pixel_distance * 1.31  # Tỷ lệ 0.76 pixel/mm<dao nguoc no deeeee>
            
            # Hiển thị chiều dài thực tế
            print(f"Khoảng cách giữa hai điểm là: {pixel_distance:.2f} pixels")
            print(f"Chiều dài thực tế: {real_distance:.2f} mm")

# Đọc ảnh từ file
img = cv2.imread('/home/ducanh/FOD/cam0/check/check40.jpeg')  # Thay đổi đường dẫn ảnh của bạn
points = []  # Danh sách lưu các điểm click chuột

# Hiển thị ảnh và xử lý sự kiện click chuột
cv2.imshow("Image", img)
cv2.setMouseCallback("Image", click_event)

cv2.waitKey(0)  # Đợi người dùng nhấn phím
cv2.destroyAllWindows()
