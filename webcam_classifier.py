import ipdb
import numpy as np
import cv2
face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')

labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

vid_capture = cv2.VideoCapture(0)


while True:
    _, frame = vid_capture.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        # reshape to 48 x 48 for our detector
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)


        # only predict when there is something to predict
        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = np.expand_dims(roi,axis=0)
            # HERE WE MUST PREDICT USING THE MODEL to get output
            # FOR THE LABEL JUST SET IT TO THE PREDICTED EMOTION FORM MODEL for ex label = labels[argmax(output)]
            label= "test"

            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_ITALIC,1,(255,0,0),2)

    cv2.imshow('Facial Emotion Detector',frame)
    #press x to quit
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

vid_capture.release()
cv2.destroyAllWindows()