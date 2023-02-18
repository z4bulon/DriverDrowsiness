import cv2
import os
from keras.models import load_model
import numpy as np
import time
face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
lefteye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
righteye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')
labels =['Closed','Opened']
model = load_model('model/Detectionmodel.h5')
path = os.getcwd()
framecapture = cv2.VideoCapture(0)
if framecapture.isOpened():
    framecapture = cv2.VideoCapture(0)
if not framecapture.isOpened():
    raise IOError("Cannot open webcam")
count = 0
closedtime = 0
border = 3
rightpred=[99]
leftpred=[99]
while(True):
    ret, frame = framecapture.read()
    height,width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_coor = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.2,minSize=(25,25))
    left_coor = lefteye.detectMultiScale(gray)
    right_coor = righteye.detectMultiScale(gray)
    for (x,y,w,h) in faces_coor:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )
    for (x,y,w,h) in right_coor:
        right=frame[y:y+h,x:x+w]
        count += 1
        right = cv2.cvtColor(right,cv2.COLOR_BGR2GRAY)
        right = cv2.resize(right,(24,24))
        right = right/255
        right =  right.reshape(24,24,-1)
        right = np.expand_dims(right, axis=0)
        rightpred = np.argmax(model.predict(right), axis=-1)
        if(rightpred[0] == 1):
            labels = 'Opened'
        if(rightpred[0]==0):
            labels = 'Closed'
        break
    for (x,y,w,h) in left_coor:
        left=frame[y:y+h,x:x+w]
        count += 1
        left = cv2.cvtColor(left,cv2.COLOR_BGR2GRAY)
        left = cv2.resize(left,(24,24))
        left = left/255
        left = left.reshape(24,24,-1)
        left = np.expand_dims(left, axis=0)
        leftpred = np.argmax(model.predict(left), axis=-1)
        if(leftpred[0] == 1):
            labels ='Opened'
        if(leftpred[0] == 0):
            labels ='Closed'
        break
    if(rightpred[0] == 0 and leftpred == 0):
        closedtime += 1
    else:
        closedtime -= 1
    if (closedtime < 0):
        closedtime = 0
    cv2.putText(frame, 'Time when eyes are closed:' + str(closedtime), (150, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1,cv2.LINE_AA)
    if (closedtime > 15):
        cv2.putText(frame, 'STOP THE CAR', (150, height-100), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255), 3,cv2.LINE_AA)
        cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
        if (border < 16):
            border = border + 2
        else:
            border = border - 2
            if (border < 2):
                border = 2
        cv2.rectangle(frame, (0, 0), (width, height), (255, 255, 255), border)
    cv2.imshow('Drowsiness detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
framecapture.release()
cv2.destroyAllWindows()
