#chuyen doi video su dung tat ca ma tran da tim duoc
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load camera matrix and distortion coefficients
with open("/home/ducanh/FOD/cam0/cameraMatrix.pkl", "rb") as f:
    camera_matrix_data = pickle.load(f)

with open("/home/ducanh/FOD/cam0/dist.pkl", "rb") as f:
    distortion_data = pickle.load(f)

with open("/home/ducanh/FOD/cam0/perspective_matrix.pkl", "rb") as f:
    perspective_matrix = pickle.load(f)  # Load ma trận chuyển đổi phối cảnh từ file

mtx = camera_matrix_data
dist = distortion_data[0]
# img_size=[640,360]
img_size=[800,450]




# Define the video input and output paths
input_video_path = '/home/ducanh/FOD/cam0/videos/output.webm'
output_video_path = '/home/ducanh/FOD/cam0/videos/output/video_1.webm'

# Open the video file
cap = cv2.VideoCapture(input_video_path)

# Get the video properties
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
# # frame_width = 200
# # frame_height = 200
# fps = int(cap.get(5))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'VP80')
# out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
out = cv2.VideoWriter(output_video_path, fourcc, fps, (img_size[0], img_size[1]))

# Process each frame in the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # # Resize the frame before applying perspective transform
    frame_resized = cv2.resize(frame, (img_size[0], img_size[1]))

    # Khử biến dạng frame trước khi áp dụng chuyển đổi phối cảnh
    undist = cv2.undistort(frame, mtx, dist, None, mtx)

    # Áp dụng ma trận chuyển đổi phối cảnh đã lưu
    warped = cv2.warpPerspective(undist, perspective_matrix, (img_size[0], img_size[1]), flags=cv2.INTER_LINEAR)
 
  # If the video is upside down, rotate it by 180 degrees
    # rotated_warped = cv2.rotate(warped, cv2.ROTATE_180)

#  # Resize the warped frame to match the output video size
#     warped_resized = cv2.resize(warped, (img_size[0], img_size[1]))

    # Ghi frame đã xử lý vào file đầu ra
    out.write(warped)
    # Process the frame using the perspective transform
    # processed_frame, _ = corners_unwarp(frame, mtx, dist)
    # processed_frame, _ = auto_perspective_transform(frame, mtx, dist)
    # print(processed_frame.shape)
    # print(type(processed_frame))
    # # Write the processed frame to the output video
    # out.write(processed_frame)

# Release the video capture and writer objects
cap.release()
out.release()

# Display a message indicating the processing is complete
print("Video processing complete.")
