# -*- coding: utf-8 -*-
"""
Created on Sat May  5 14:04:31 2018

@author: balag
"""
#Please put the following files in the same folder or provide the appropriate path :
# haarcascade_frontalface_default.xml, haarcascade_nose.xml,haarcascade_smile.xml,haarcascade_eye.xml
#Install open cv  and run the code
import cv2
#Loading the cascades
face_cascade = cv2.CascadeClassifier('C:/Users/balag/OneDrive/Desktop/project/Computer_Vision_A_Z_Template_Folder\Module 1 - Face Recognition/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:/Users/balag/OneDrive/Desktop/project/Computer_Vision_A_Z_Template_Folder\Module 1 - Face Recognition/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('C:/Users/balag/OneDrive/Desktop/project/Computer_Vision_A_Z_Template_Folder\Module 1 - Face Recognition/haarcascade_smile.xml')
nose_cascade=cv2.CascadeClassifier('C:/Users/balag/OneDrive/Desktop/project/Computer_Vision_A_Z_Template_Folder\Module 1 - Face Recognition/haarcascade_nose.xml')
#Defining a function that will do detection
def detect(gray,frame):# We create a function that takes as input the image in black and white (gray) and the original image (frame), and that will return the same image with the detector rectangles. 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)# We apply the detectMultiScale method from the face cascade to locate one or several faces in the image.
    
    for(x,y,w,h) in faces:# For each detected face:
        cv2.rectangle(frame,(x,y),(x+w,y+h), (255,0,0), 2)# We paint a rectangle around the face.
        roi_gray=gray[y:y+h,x:x+w]# We get the region of interest in the black and white image.
        roi_color=frame[y:y+h,x:x+w]# We get the region of interest in the colored image.
        
        eye=eye_cascade.detectMultiScale(roi_gray, 1.1, 22)# We apply the detectMultiScale method to locate one or several eyes in the image.
        for(ex,ey,ew,eh) in eye:# For each detected eye:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh), (0,255,0), 2)# We paint a rectangle around the eyes, but inside the referential of the face.
        
        nose=nose_cascade.detectMultiScale(roi_gray, 1.1, 30)# We apply the detectMultiScale method to locate nose in the image.
        for(nx,ny,nw,nh) in nose:# For each detected nose:
            cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh), (0,100,0), 2)# We paint a rectangle around the nose, but inside the referential of the face.
        
        smile=smile_cascade.detectMultiScale(roi_gray, 1.7, 30)# We apply the detectMultiScale method to detect smile.
        for(sx,sy,sw,sh) in smile:# For each detected smile:
            num=0#defining the name attached to the image file
            cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh), (0,0,255), 2)#We paint a rectangle around the around the mouth if we see a smile, but inside the referential of the face.
            cv2.imwrite('opencv'+str(num)+'.jpg',frame)#converting the output to jpeg
            num=num+1
    return frame# We return the image with the detector rectangles.

#Doing Face detection using Webcam
video_capture=cv2.VideoCapture(0)# We turn the webcam on.
while True:# We repeat infinitely (until break):
    _,frame=video_capture.read() # We get the last frame.
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)# We do some colour transformations.
    canvas=detect(gray,frame)# We get the output of our detect function.
    cv2.imshow('Video',canvas)# We display the outputs.
    if cv2.waitKey(1) & 0xFF == ord('q'):# If we type on the keyboard:
        break# We stop the loop.
video_capture.release()# We turn the webcam off.
cv2.destroyAllWindows()  # We destroy all the windows inside which the images were displayed.
    