'''
Author: SingleBiu
Date: 2024-10-17 15:12:35
LastEditors: SingleBiu
LastEditTime: 2024-10-17 16:08:51
Description: file content
'''
'''
Author: SingleBiu
Date: 2024-10-17 15:12:35
LastEditors: SingleBiu
LastEditTime: 2024-10-17 15:28:05
Description: A demo for face detect
'''
import os
#pip install opencv-contrib-python
import cv2 as cv
from PIL import Image
import numpy as np

def getImageAndLabels(path):
    faceSamples = []
    ids =[]
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    face_detector = cv.CascadeClassifier('./classifier/lbpcascade_frontalface.xml')
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img,'uint8')
        faces = face_detector.detectMultiScale(img_numpy)
        id = int(os.path.split(imagePath)[1].split('.')[0])
        for x,y,w,h in faces:
            ids.append(id)
            faceSamples.append(img_numpy[y:y+h,x:x+w])
        print('id',id)
        print('fs:',faceSamples)
        return faceSamples,ids

if __name__ ==  '__main__':
    path = "./data/"
    faces,ids = getImageAndLabels(path)
    recognizer = cv.face.LBPHFaceRecognizer_create()
    recognizer.train(faces,np.array(ids))
    recognizer.write('trainer/trainer.yml')