import cv2 as cv
import matplotlib.pyplot as plt
from my_template import *

# Read image
img = cv.imread('.\sign/sign.jpg',1)


template = cv.imread(('.\sign/tem5.jpg'),1)
h,w = template.shape[:2]

# Use normalized cross correlation (ncc)
meth = 'cv.TM_CCOEFF_NORMED'
method = eval(meth)

# Apply template matching
# res = cv.matchTemplate(img,template,method)
res = myTemplate(img,template)
cv.normalize(res,res,0,1,cv.NORM_MINMAX,-1)
min_val, max_val, min_loc,max_loc = cv.minMaxLoc(res)

# Draw rectangle
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv.rectangle(img,top_left,bottom_right,255,5)
#cv.rectangle(res,top_left,bottom_right,255,5)

# Plot
plt.subplot(121)
plt.imshow(res,cmap='gray')
plt.title('Correlation Result')
plt.xticks([]),plt.yticks([])

plt.subplot(122)
plt.axis("off")
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)) # OpenCV는 RGB가 아니라 BGR로 읽어온다
plt.title('Detected Point')
plt.xticks([]), plt.yticks([])
plt.show()

cv.namedWindow('template', flags=cv.WINDOW_NORMAL)
cv.imshow('template',template)
cv.resizeWindow('template',250,250)
cv.waitKey(0)
cv.destroyAllWindows()