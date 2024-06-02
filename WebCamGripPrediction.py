from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import imutils
import os
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.callbacks import ModelCheckpoint
import keras
from keras.applications import VGG16 #VGG class
from keras.applications import DenseNet121
from keras.applications import ResNet101 #resnet
from keras.layers import Conv2D, MaxPool2D, InputLayer, BatchNormalization #CNN and alexnet classes

main = tkinter.Tk()
main.title("Deep Grip Prediction from Webcam")
main.geometry("1300x1200")

global filename
global cnn_model

bg = None
playcount = 0

names = ['Arm', 'Carrom', 'Flipper', 'Googly', 'LegBreak', 'Swing']

bgModel = cv2.createBackgroundSubtractorMOG2(0, 50)

def remove_background(frame):
    fgmask = bgModel.apply(frame, learningRate=0)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

#now training CNN algorithm on GRIP dataset and evaluating its performnace in terms of accuracy, precision, recall 
cnn_model = Sequential()
#defining input shape layer
cnn_model.add(InputLayer(input_shape=(32, 32, 3)))
#defining CNN layer with 25 filters to filter images 25 time3
cnn_model.add(Conv2D(25, (5, 5), activation='relu', strides=(1, 1), padding='same'))
cnn_model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
cnn_model.add(Conv2D(50, (5, 5), activation='relu', strides=(2, 2), padding='same'))
cnn_model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
cnn_model.add(BatchNormalization())
cnn_model.add(Conv2D(70, (3, 3), activation='relu', strides=(2, 2), padding='same'))
cnn_model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
cnn_model.add(BatchNormalization())
cnn_model.add(Flatten())
cnn_model.add(Dense(units=100, activation='relu'))#definining output layer
cnn_model.add(Dense(units=100, activation='relu'))
cnn_model.add(Dropout(0.25))
cnn_model.add(Dense(units=len(names), activation='softmax'))#defining prediction layer
#compiling, training and loading model
cnn_model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
cnn_model.load_weights("model/cnn_weights.hdf5")

def run_avg(image, aWeight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return
    cv2.accumulateWeighted(image, bg, aWeight)

def segment(image, threshold=25):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    ( cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return
    else:
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)


def webcamPredict():
    global playcount
    oldresult = 'none'
    count = 0
    fgbg2 = cv2.createBackgroundSubtractorKNN(); 
    aWeight = 0.5
    camera = cv2.VideoCapture(0)
    top, right, bottom, left = 10, 350, 325, 690
    num_frames = 0
    while(True):
        (grabbed, frame) = camera.read()
        frame = imutils.resize(frame, width=700)
        frame = cv2.flip(frame, 1)
        clone = frame.copy()
        (height, width) = frame.shape[:2]
        roi = frame[top:bottom, right:left]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (41, 41), 0)
        if num_frames < 30:
            run_avg(gray, aWeight)
        else:
            temp = gray
            hand = segment(gray)
            if hand is not None:
                (thresholded, segmented) = hand
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                roi = frame[top:bottom, right:left]
                roi = fgbg2.apply(roi); 
                cv2.imwrite("test.jpg",roi)
                img = cv2.imread("test.jpg")
                img = cv2.resize(img, (32, 32))
                img = img.reshape(1, 32, 32, 3)
                img = np.array(img, dtype='float32')
                img /= 255
                predict = cnn_model.predict(img)
                value = np.amax(predict)
                cl = np.argmax(predict)
                result = names[np.argmax(predict)]
                if value >= 0.90:
                    print(str(value)+" "+str(result))
                    cv2.putText(clone, 'Deep Grip Preducted as : '+str(result), (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 255), 2)                    
                cv2.imshow("video frame", roi)
            else:
                cv2.putText(clone, 'No Motion', (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 255), 2)
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)
        num_frames += 1
        cv2.imshow("Video Feed", clone)
        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord("q"):
            break
    camera.release()
    cv2.destroyAllWindows()    
    

    
    
font = ('times', 16, 'bold')
title = Label(main, text='Deep Grip Prediction from Webcam',anchor=W, justify=CENTER)
title.config(bg='yellow4', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)
font1 = ('times', 13, 'bold')

predictButton = Button(main, text="Deep Grip Prediction from Webcam", command=webcamPredict)
predictButton.place(x=50,y=100)
predictButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=15,width=78)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=450,y=100)
text.config(font=font1)

main.config(bg='magenta3')
main.mainloop()
