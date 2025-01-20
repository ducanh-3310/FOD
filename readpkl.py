import pickle

# Đọc file pickle
with open('/home/ducanh/FOD/cam0/cameraMatrix.pkl', 'rb') as file:  # Mở file với chế độ đọc nhị phân ('rb')
    data = pickle.load(file)

# Hiển thị nội dung dữ liệu đã được tải
print(data)

# import numpy as np

# # Đọc file .npy
# data_npy = np.load('/home/ducanh/FOD/cam0/cameraMatrix.npy')

# # Hiển thị nội dung từ file .npy
# print(data_npy)


# import numpy as np
# import pickle

# # Đọc file .pkl
# with open('/home/ducanh/FOD/cam0/perspective_matrix.pkl', 'rb') as f:
#     data = pickle.load(f)

# # Kiểm tra nếu dữ liệu là mảng NumPy
# if isinstance(data, np.ndarray):
#     # Lưu dưới dạng .npy
#     np.save('/home/ducanh/FOD/cam0/perspective_matrix.npy', data)
# else:
#     print("Dữ liệu trong file .pkl không phải là mảng NumPy. Hãy kiểm tra lại!")
