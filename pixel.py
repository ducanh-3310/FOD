# Tinh xem anh chup duoc la bao nhieu pixel
import cv2

def analyze_image(image_path):
    # Đọc ảnh từ file
    image = cv2.imread(image_path)
    
    if image is None:
        print("Không thể đọc được ảnh từ đường dẫn.")
        return
    
    # Lấy độ phân giải ảnh
    height, width = image.shape[:2]
    total_pixels = height * width
    print(f"Độ phân giải ảnh: {width} x {height} ({total_pixels} pixels)")

    # Nhận xét chất lượng ảnh dựa vào tổng số pixel
    if total_pixels >= 8000000:
        quality = "Cao (8MP trở lên)"
    elif total_pixels >= 2000000:
        quality = "Trung bình (2MP - 8MP)"
    else:
        quality = "Thấp (dưới 2MP)"
    print(f"Chất lượng ảnh: {quality}")

    # Hiển thị ảnh
    cv2.imshow('Ảnh đã chọn', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Đường dẫn đến ảnh
image_path = "/home/ducanh/FOD/cam0/check/check10.jpeg"  # Thay bằng đường dẫn ảnh của bạn
analyze_image(image_path)


# import cv2

# def get_video_resolution_and_pixel_count(video_path):
#     # Mở video bằng OpenCV
#     cap = cv2.VideoCapture(video_path)
    
#     if not cap.isOpened():
#         raise ValueError("Không thể mở video.")

#     # Lấy kích thước khung hình của video
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     # Tính số pixel trong mỗi khung hình
#     num_pixels = frame_width * frame_height

#     # Trả về kích thước khung hình và số pixel
#     return (frame_width, frame_height, num_pixels)

# # Đường dẫn video
# video_path = '/home/ducanh/Downloads/video_processed.mp4'

# # Gọi hàm và in kết quả
# frame_width, frame_height, num_pixels = get_video_resolution_and_pixel_count(video_path)
# print(f"Resolution: {frame_width}x{frame_height}")
# print(f"Number of pixels per frame: {num_pixels}")
