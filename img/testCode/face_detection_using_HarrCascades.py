'''
Created on 11-Jan-2019
@author: http://www.eduliant.com
'''


import cv2


''' 
    Step 1. Load the pre-trained xml classifiers available at
    https://github.com/opencv/opencv/tree/master/data/haarcascades
'''

cascade_face = cv2.CascadeClassifier("/home/rajnish.kumar/opencv4/opencv/data/haarcascades/haarcascade_frontalface_default.xml")

'''
    Step 2. Load Our test Image (Also video) in grayscale mode
'''

normalimg = cv2.imread("/home/rajnish.kumar/Downloads/download.jpeg")
grayimg = cv2.cvtColor(normalimg,cv2.COLOR_BGR2GRAY)


'''
    Step 3. if test image (or video) contains faces then it will return the positions of detected faces as Rect(x,y,w,h)
    Using these locations we can create ROI (what is ROI is discussed later).
'''

faces = cascade_face.detectMultiScale(grayimg,1.3,5) # what is 1.3,5 we will discuss it later
for (x,y,w,h) in faces:
    cv2.rectangle(normalimg,(x,y),(x+w,y+h),(0,0,0),2)
    
   
cv2.imshow("ImageWithFaceDetected",normalimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
