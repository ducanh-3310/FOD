import cv2

cap = cv2.VideoCapture('/home/ducanh/FOD/cam0/videos/vd2.webm')
# cap = cv2.VideoCapture(0)
num = 15

while cap.isOpened():

    succes, img = cap.read()

    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('test/img' + str(num) + '.jpeg', img)
        print("image saved!")
        num += 1

    cv2.imshow('Img',img)

# Release and destroy all windows before termination
cap.release()

cv2.destroyAllWindows()