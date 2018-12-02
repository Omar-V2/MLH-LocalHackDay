import cv2
import os
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
font = cv2.FONT_HERSHEY_SIMPLEX

def generate_gesture(ges_name, num_train_samples, save=True):
    cam = cv2.VideoCapture(0)
    x, y, w, h = (160, 100, 300, 300)

    clf = load_model('trained.h5')
    # clf = load_model('trained2.h5')
    mapper = {5: "One!", 3: "Five!", 4: "Three!", 7: "Two!", 6:"Two!"}

    while True:
        ret, frame = cam.read() # frame is unaltered webcam feed
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert webcam feed to grayscale
        _, thresh = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) # apply some filtering
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
        ROI = thresh[y: y+h, x: x+w]
        cv2.imshow('thresh', ROI)
        cv2.waitKey(1)
        # print(ROI.shape)
        im_in = cv2.imwrite('predictor.jpg', ROI)
        im_in = load_img('predictor.jpg')
        im_in = im_in.resize((28, 28), Image.NEAREST)
        im_in = img_to_array(im_in)
        im_in = np.expand_dims(im_in, axis=0)
        pred = clf.predict_classes(im_in)[0]
        cv2.putText(frame, mapper.get(pred, 'not sure'), (230, 200), font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        # cv2.putText(frame, str(pred), (230, 200), font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Input', frame)



        if cv2.waitKey(1) and 0xFF == ord('q'):
            break



    cam.release()
    cv2.destroyAllWindows()



generate_gesture('five', 500, save=True)