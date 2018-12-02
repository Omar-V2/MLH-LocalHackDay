import cv2
import os
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


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
        cv2.waitKey(1)
        # print(ROI.shape)
        im_in = cv2.imwrite('predictor.jpg', ROI)
        im_in = load_img('predictor.jpg')
        im_in = im_in.resize((150, 150), Image.NEAREST)
        im_in = img_to_array(im_in)
        im_in = np.expand_dims(im_in, axis=0)
        print(clf.predict(im_in))



        if cv2.waitKey(1) and 0xFF == ord('q'):
            break



    cam.release()
    cv2.destroyAllWindows()



generate_gesture('five', 500, save=True)