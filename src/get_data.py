import cv2
import os
import shutil
import time






def generate_gesture(ges_name, num_train_samples, save=True):
    cam = cv2.VideoCapture(0)
    x, y, w, h = (120, 100, 300, 300)
    
    i = 0
    j = 0

    train_dir = 'data/train/{}/'.format(ges_name)
    val_dir = 'data/validation/{}/'.format(ges_name)
    if not os.path.exists(train_dir) and not os.path.exists(val_dir):
        os.mkdir(train_dir)
        os.mkdir(val_dir)

    while True:
        ret, frame = cam.read() # frame is unaltered webcam feed
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert webcam feed to grayscale
        _, thresh = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) # apply some filtering
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
        ROI = thresh[y: y+h, x: x+w]
        cv2.imshow('Input', frame)
        cv2.imshow('thresh', ROI)
        cv2.waitKey(1)
        if j > 40 and save:
            # time.sleep(0.1)
            cv2.imwrite(train_dir+'{}.jpg'.format(i), ROI)
            print("Saved train image {}".format(i))
            i += 1
        else:
            print(j)
            j += 1 
        if cv2.waitKey(1) and 0xFF == ord('q') or i > num_train_samples:
            break



    cam.release()
    cv2.destroyAllWindows()



generate_gesture('five', 500, save=True)
