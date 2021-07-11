import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, time, ssl


if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)): 
  ssl._create_default_https_context = ssl._create_unverified_context

X = np.load('image.npz')['arr_0']
y = pd.read_csv('labels.csv')['labels']
print(pd.Series(y).value_counts())

classes = ['A', 'B', 'C', 'D', 'E','F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
no_of_classes = len(classes)


x_train, x_test, y_train, y_test = train_test_split(X, y, random_state = 9, train_size = 7500, test_size = 2500)
x_train_scaled = x_train/255.0
x_test_scaled = x_test/255.0

clf = LogisticRegression(solver='saga', multi_class='multinomial').fit(x_train_scaled, y_train)


y_pred = clf.predict(x_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)


cam = cv2.VideoCapture(0)

while(True):
  try: 
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    upperleft = (int(width/2-56), int(height/2-56))
    bottomright = (int(width/2+56), int(height/2+56))
    cv2.rectangle(gray, upperleft, bottomright, (0, 255, 0), 2)
    roi = gray[upperleft[1]:bottomright[1], upperleft[0]:bottomright[0]]

    impil = Image.fromarray(roi)
    imgbw = impil.convert('L')
    imgbwresize = imgbw.resize((28, 28), Image.ANTIALIAS)
    imgbwresizein = PIL.ImageOps.invert(imgbwresize)
    pixelfilter = 20
    min_pixel = np.percentile(imgbwresizein, pixelfilter)
    imgbwresizeinscaled = np.clip(imgbwresizein-min_pixel, 0, 255) 
    max_pixel = np.max(imgbwresizein) 
    imgbwresizeinscaled = np.asarray(imgbwresizeinscaled)/max_pixel 
    
    test_sample = np.array(imgbwresizeinscaled).reshape(1,784) 
    test_pred = clf.predict(test_sample) 
    print("Predicted class is: ", test_pred)

    cv2.imshow('frame', gray)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
      break 
  
  except Exception as e: 
    pass
    

cam.release() 
cv2.destroyAllWindows()