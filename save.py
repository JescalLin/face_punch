#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import face_recognition
name = "Yuhao"
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

i = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame_or = frame.copy()
    face_locations = face_recognition.face_locations(frame, model='cnn')

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break


    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    # 顯示圖片
    cv2.imshow('frame', frame)
    # 按下 q 鍵離開迴圈
    if cv2.waitKey(1) == ord('q'):
        i = i +1
        cv2.imwrite("./com_img/"+name+"_"+str(i)+".jpg",frame_or)

        

# 釋放該攝影機裝置
cap.release()
cv2.destroyAllWindows()