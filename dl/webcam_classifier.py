import ipdb
import numpy as np
import cv2
import torch
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
labels = ['Angry','Disgust','Fear','Happy','Sad', 'Surprise', 'Neutral']

vid_capture = cv2.VideoCapture(0)

def video_capture(model):
    while True:
        _, frame = vid_capture.read()
        
        # ipdb.set_trace()
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
                # roi to pytorch tensor

                roi = torch.from_numpy(roi).float()
                roi = roi.unsqueeze(0)
                # ipdb.set_trace()
                prediction = model(roi)
                
                max_idx = torch.argmax(prediction)
                # tensor to number
                idx = max_idx.item()
                # ipdb.set_trace()
                label = labels[idx]
                # ipdb.set_trace()
                # label= "test"

                label_position = (x,y)
                cv2.putText(frame,label,label_position,cv2.FONT_ITALIC,1,(255,0,0),2)

        cv2.imshow('Facial Emotion Detector',frame)
        #press x to quit
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

    vid_capture.release()
    cv2.destroyAllWindows()