import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

def myTemplate(img,tem):
    W,H = img.shape[0:2]
    w,h = tem.shape[0:2]
    res = np.zeros((W-w+1, H-h+1))
    
    # template 
    t_diff = tem - np.mean(tem) # t-t_bar
    t_var = np.sqrt(np.sum(np.square(t_diff))) # (∑((t-t_bar)^2))^0.5
     
    # image

    for x in range(0,W-w+1) :
        print(x)
        for y in range(0,H-h+1) :
            conv = 0
            img_mean = np.mean(img[x:x+w,y:y+h]) # f_bar
            img_diff = img[x:x+w,y:y+h] - img_mean # f - f_bar
            img_var = np.sqrt(np.sum(np.square(img_diff))) # (∑((f-f_bar)^2))^0.5
            conv = conv + np.sum(np.multiply(t_diff,img_diff)) # convolution
            
            res[x,y] = conv / (img_var*t_var)
    return res