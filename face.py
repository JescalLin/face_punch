import face_recognition
import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, isdir, join
import time
from datetime import datetime
from PIL import ImageFont, ImageDraw, Image


video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)


known_face_encodings = []
known_face_names = []
face_flag= []

allFileList = os.listdir("com_img/")

for file in allFileList:
    if 'JPG' in file or 'jpg' in file:
        known_face_names.append(file)

for i in range(len(known_face_names)):
    img = cv2.imread('com_img/'+known_face_names[i])
    face_encoding = face_recognition.face_encodings(img)[0]
    known_face_encodings.append(face_encoding)

fontPath = "C:\\WINDOWS\\Fonts\\kaiu.TTF"
font = ImageFont.truetype(fontPath, 40)
flag = 0

while True:
    
    ret, frame = video_capture.read()


    cv2.rectangle(frame, (0, 0), (200, 25), (0, 0, 0), cv2.FILLED)

    sys_time = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    cv2.putText(frame, sys_time , (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame, model='cnn')
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index].split('_')[0]

        if(face_distances[best_match_index]>0.4):
            name = "Unknown"

        
        value = str((1-float(face_distances[np.argmin(face_distances)]))*100)[:4]+"%"
        if name !="Unknown":
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom + 40), (right, bottom), (0, 255, 0), cv2.FILLED)
            flag = flag + 1

        
            if flag>7:
                cv2.rectangle(frame, (left, bottom + 40), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.8,(255, 255, 255), 1)
                cv2.putText(frame, value, (left + 6, bottom + 40), cv2.FONT_HERSHEY_DUPLEX, 0.8,(255, 255, 255), 1)
            if flag>35:
                frame = np.zeros((480, 640, 3), np.uint8)
                text = "\n\n\n\n"+name+":打卡成功! \n"+sys_time
                imgPil = Image.fromarray(frame)
                draw = ImageDraw.Draw(imgPil)
                draw.text((120, 10), text, font=font, fill=(255, 255, 255))
                frame = np.array(imgPil)
            if flag==36:
                #存資料庫
                pass
            if flag==60:
                flag = 0




        else:
            flag = 0
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom + 40), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.8,(255, 255, 255), 1)
            cv2.putText(frame, value, (left + 6, bottom + 40), cv2.FONT_HERSHEY_DUPLEX, 0.8,(255, 255, 255), 1)


    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()