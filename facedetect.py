'''
Author: SingleBiu
Date: 2024-10-17 13:43:08
LastEditors: SingleBiu
LastEditTime: 2024-10-17 14:42:20
Description: A demo for face detect
'''
import cv2 as cv

def face_detect_method():
    grey_img = cv.cvtColor(img_resized,cv.COLOR_BGRA2GRAY)
    face_detector = cv.CascadeClassifier('./classifier/lbpcascade_frontalface.xml')
    face = face_detector.detectMultiScale(grey_img,1.1,5,0,(10,10),(200,200))
    for x,y,w,h in face:
        cv.rectangle(img_resized,(x,y),(x+w,y+h),color=(0,0,255),thickness=2)
    cv.imshow('result',img_resized)
    
#img = cv.imread("./face.jpg")
img = cv.imread("./face2.jpg")
img_resized = cv.resize(img,(400,600))
face_detect_method()

while True:
    if ord('m') == cv.waitKey(0):
        break

cv.destroyAllWindows()