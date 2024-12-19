import cv2

# Đường dẫn đến video gốc và video đầu ra
input_video_path = '/home/ducanh/FOD/cam0/videos/vd1.webm'
output_video_path = '/home/ducanh/FOD/cam0/videos/output/output.webm'

# Mở video gốc
cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print("Không thể mở video.")
    exit()

# Lấy thông tin về video (kích thước khung hình và FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Kiểm tra nếu video có độ phân giải 1920x1080
print(f"Video Resolution: {frame_width}x{frame_height}")

# Đặt kích thước mới cho video đầu ra
output_width = 800
output_height = 450

# Tạo đối tượng VideoWriter để ghi video đầu ra
fourcc = cv2.VideoWriter_fourcc(*'VP80')  # Chọn codec VP8 cho WebM
out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))

# Đọc và thay đổi kích thước từng khung hình của video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Thay đổi kích thước khung hình về 800x450
    resized_frame = cv2.resize(frame, (output_width, output_height))

    # Ghi khung hình đã thay đổi kích thước vào video đầu ra
    out.write(resized_frame)

# Giải phóng các đối tượng video
cap.release()
out.release()

print("Video đã được chuyển đổi và lưu lại.")
