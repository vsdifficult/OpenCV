import cv2, numpy, imutils, easyocr 
from matplotlib import pyplot as pl

img = cv2.imread("w222-mercedes.jpg") 

number = cv2.CascadeClassifier("models/haarcascade_russian_plate_number.xml") 

detect = number.detectMultiScale(img, scaleFactor=2, 
                                 minNeighbors=1) 

gray = cv2.cvtColor(img, cv2.COLOR_BAYER_BGGR2GRAY) 

img_filter = cv2.bilateralFilter(gray, 11, 15, 15) 
edges = cv2.Canny(img_filter, 30, 200  )


count = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
cont = imutils.grab_contours(count)   
cont = sorted(cont, key=cv2.contourArea, reverse=True)[:8] 

pos = None 

for c in cont: 
    aprox = cv2.approxPolyDP(c, 10, True) 

    if len(aprox) == 4: 
        pos = aprox 
        break  

mask = numpy.zeros(gray.shape, numpy.uint8)
new_img = cv2.drawContours(mask, [pos], 0, 255, -1) 
biwi_img = cv2.bitwise_and(img, img, mask=mask) 

x, y = numpy.where(mask == 255)
x1, y1 = numpy.min(x), numpy.min(y) 
x2, y2 = numpy.max(x), numpy.max(y) 

crop = gray[x1:x2, y1:y2] 


text = easyocr.Reader(['en']) 
text = text.readtext(crop)

print(text)


""""""
pl.imshow(cv2.cvtColor(crop, cv2.COLOR_RGBA2GRAY)) 
pl.show()



"""for (x, y, w, h) in detect: 
    cv2.rectangle(img, (x, y), (x + w, y + h), thickness=2) 
"""
