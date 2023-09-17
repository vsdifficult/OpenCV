import cv2 

"""img = cv2.imread("imageee.jpeg") 
new_imge = cv2.resize(img, (1000, 100))
cv2.imshow("Result", new_imge)   


print(new_imge.shape)
cv2.waitKey(0) 
"""

import numpy as np  
#import imutils
#from matplotlib import pyplot as pl


"""
photo = np.zeros((500, 500, 3), dtype="uint8") 
print(photo) 

#BGR

#photo[:] = 2, 45, 5 # color picker 
#photo.shape
cv2.rectangle(photo, (0,0) (100, 100))

cv2.imshow("Photo", photo) 
cv2.waitKey(0) 
""" 

img = cv2.imread("hu.jpeg") 
#img = cv2.cvtColor(img, cv2.COLOR_BAYER_BGGR2GRAY) 

smile = cv2.CascadeClassifier("models/haarcascade_smile.xml") 
human = cv2.CascadeClassifier("models/haarcascade_fullbody.xml")
eye = cv2.CascadeClassifier("models/haarcascade_eye.xml")
face_dt = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")


smiles = smile.detectMultiScale(img, scaleFactor=3,  
                                  minNeighbors=3) 

_human = human.detectMultiScale(img, scaleFactor=2, 
                               minNeighbors=3)

eyes = eye.detectMultiScale(img, scaleFactor=3, 
                            minNeighbors=2)  

face = face_dt.detectMultiScale(img, scaleFactor=2, 
                                minNeighbors=2)


for (x, y, w, h) in smiles:  
         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)  
         for (x, y, w, h) in eyes: 
                 cv2.rectangle(img, (x, y), (x +w, y + h),(255, 0, 0), thickness=2)

for (x, y, w, h) in _human: 
      cv2.rectangle(img, (x, y), (x +w, y + h),(5, 250, 0), thickness=2) 
      for (x, y, w, h) in face: 
                  cv2.rectangle(img, (x, y), (x +w, y + h),(50, 25, 5), thickness=2) 

cv2.imshow("Result", img) 
cv2.waitKey(0)  


