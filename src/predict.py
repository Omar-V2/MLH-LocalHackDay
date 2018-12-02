import cv2
import os
import numpy as np
from keras.models import load_model

def generate_gesture(ges_name, num_train_samples, save=True):
    cam = cv2.VideoCapture(0)
    x, y, w, h = (120, 100, 300, 300)

    clf = load_model('trained.h5')

    while True:
        ret, frame = cam.read() # frame is unaltered webcam feed
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert webcam feed to grayscale
        _, thresh = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) # apply some filtering
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
        ROI = thresh[y: y+h, x: x+w]
        cv2.imshow('Input', frame)
        cv2.imshow('thresh', ROI)
        # model_in = cv2.resize(ROI, (128, 128))
        model_in = np.reshape(ROI, (-1, 156, 156, 3))
        cv2.waitKey(1)
        clf.predict(model_in)

        if cv2.waitKey(1) and 0xFF == ord('q'):
            break



    cam.release()
    cv2.destroyAllWindows()



generate_gesture('five', 500, save=True)
