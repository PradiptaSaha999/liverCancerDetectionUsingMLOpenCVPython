# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 02:40:23 2017

@author: Pradipta
"""

import cv2
import numpy as np
from sklearn.externals import joblib
from skimage.feature import hog
from sklearn.svm import LinearSVC
from collections import Counter




def printValue(event,x,y,flags,param):
##    [b,g,r]=img[y,x]
##    [h,s,v]=hsv[y,x]
    print(im_gray1[y,x])


test= True
i=1
labels=[]
features=[]
while test:
    
    name="New folder/Positive/a ("+str(i)+").bmp"
    im = cv2.imread(name)  #import image
    im = cv2.GaussianBlur(im, (9, 9), 0)
    im = cv2.medianBlur(im,3)
    im_gray1 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    
#    average_color = np.average(im_gray1, axis=0)
#    im_gray1[:,:]=im_gray1[:,:]+(90-average_color.mean())
#    print(average_color.mean())
    
    lower_value = np.array([60])
    upper_value = np.array([190])
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(im_gray1, lower_value, upper_value)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(im_gray1,im_gray1, mask= mask)
    u= res[res[:,:] >0 ]
#    print(u.mean())
#    print(u.max())
#    print(u.min())

    res = cv2.medianBlur(res,11)
#    im_gray=cv2.fastNlMeansDenoising(im_gray)
    res = cv2.GaussianBlur(res, (9, 9), 0)
    
    # Threshold the image
#    ret, res = cv2.threshold(res, (u.mean()+10), 255, 0)
    res[res[:,:] < (u.mean()-30) ]=0
    res[res[:,:] > (u.mean()+20) ]=255
#    h,w,c=im.shape
    # Find contours in the image
    _, ctrs, _ = cv2.findContours(res.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areaVal=[ cv2.contourArea(ctr) for ctr in ctrs]
    maxArea=np.max(areaVal)
    ind=np.argmax(areaVal)
#    print(ind)
    # Get rectangles contains each contour
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    
    # For each rectangular region, calculate HOG features and predict
    # the digit using Linear SVM.
#    im = cv2.drawContours(im, ctrs, ind, (0,255,0), 3)
    mask = np.zeros_like(im_gray1) # Create mask where white is what we want, black otherwise
    cv2.drawContours(mask, ctrs, ind, 255, -1) # Draw filled contour in mask
    out = np.zeros_like(im_gray1) # Extract out the object and place into output imag
    out[mask == 255] = im_gray1[mask == 255]
    lower_value = np.array([60])
    upper_value = np.array([200])
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(out, lower_value, upper_value)
    # Bitwise-AND mask and original image
    out = cv2.bitwise_and(out,out, mask= mask)
    v= out[out[:,:] >0 ]
#    print(v.mean())

    out[out[:,:] < (u.mean()) ]=255
    out[out[:,:] > (u.mean()+10) ]=0
    out = cv2.medianBlur(out,5)
#    out = cv2.GaussianBlur(out, (9, 9), 0)
    
#    _, ctrsw, _ = cv2.findContours(out.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#    areaVal=[ cv2.contourArea(ctr) for ctr in ctrsw]
#    maxArea=np.max(areaVal)
#    ind=np.argmax(areaVal)
#    print(ind)
#    # Get rectangles contains each contour
##    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
#    
#    # For each rectangular region, calculate HOG features and predict
#    # the digit using Linear SVM.
#    im = cv2.drawContours(im, ctrs, -1, (0,255,0), 3)
#    mask = np.zeros_like(im_gray1)
#    
#    for ctr in ctrsw:
#        
#        
#        if cv2.contourArea(ctr) !=  maxArea:
#            mask=cv2.drawContours(mask, [ctr], 0, 255, -1)
#        
        
        
#    mask = np.zeros_like(im_gray1) # Create mask where white is what we want, black otherwise
#    cv2.drawContours(out, ctrs, -1, (0,255,0), 3) # Draw filled contour in mask
#    lasts = np.zeros_like(im_gray1) # Extract out the object and place into output imag
#    lasts[mask == 255] = im_gray1[mask == 255]
#    #im = cv2.resize(im, (int(w/2),int(h/2)))
    imq = cv2.resize(im,(512,512))
    cv2.imshow("TestImageGry", res)
    cv2.imshow("TestImage", imq)
    cv2.imshow("TestImageg", out) 
    cv2.setMouseCallback("TestImage",printValue)
    while(1):
        cv2.imshow("TestImageg", out) 
        cv2.imshow("TestImageGry", res)
        cv2.imshow("TestImage", imq)
        

        if cv2.waitKey(20) & 0xFF == 27:
            break
    i=i+1
#    inp=input()
    print(i)
    if i is 32:
        test=False   


print("Done!!!!!")
        
cv2.waitKey()
cv2.destroyAllWindows()