'''
Author: SingleBiu
Date: 2024-10-17 14:43:24
LastEditors: SingleBiu
LastEditTime: 2024-10-17 16:09:03
Description: A demo for face detect
'''
import cv2 as cv

def face_detect_method(img):
    grey_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    face_detector = cv.CascadeClassifier('./classifier/lbpcascade_frontalface.xml')
    face = face_detector.detectMultiScale(grey_img,1.12,4)
    for x,y,w,h in face:
        cv.rectangle(img,(x,y),(x+w,y+h),color=(0,0,255),thickness=2)
    cv.imshow('result',img)

#back camera
# cap = cv.VideoCapture(0)
#front camera
cap = cv.VideoCapture(1)
#read video
# cap = cv.VideoCapture("./1.mp4")

while True:
    flag,frame = cap.read()
    if not flag:
        break
    face_detect_method(frame)
    if ord('m')==cv.waitKey(1):
        break

cv.destroyAllWindows()
cap.release()