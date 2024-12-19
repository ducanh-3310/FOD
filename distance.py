# tim khoang cach va toa do giua 2 diem tren anh cho truoc
import cv2
import numpy as np

points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Tọa độ x, y:", x, y)
        points.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('image', img)

img = cv2.imread('/home/ducanh/FOD/cam0/undis_0.jpeg')
cv2.imshow('image', img)
cv2.setMouseCallback('image', click_event)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()

print("Tọa độ các điểm:", points)

if len(points) == 2:
    (x1, y1), (x2, y2) = points
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    print("Khoảng cách giữa hai điểm là:", distance)
else:
    print("Chọn sai.")
