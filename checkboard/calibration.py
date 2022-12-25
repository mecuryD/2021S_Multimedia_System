'''
# Termination Criteria
cv.TERM_CRITERIA_EPS : 정해둔 정확도에 다다르면 STOP
cv.TERM_CRITERIA_MAX_ITER : 지정한 반복 수만큼을 지나면 STOP
cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER : 위의 조건 중 하나라도 만족하면 STOP
'''

# Import Library
import glob
import cv2 as cv
import numpy as np

# Termination Criteria
# Iteration = 30, accuracy = 0.001
criteria =  (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001) 

# Object Point (3D)를 준비한다. like (0,0,0), (1,0,0),...,(6,7,0) 처럼
objp = np.zeros((6*5,3), np.float32)

# np.mgrid[0:7,0:8]으로 (2,7,8) 배열 생성
# Transpose 해줘서 (6,7,2)로, reshape(-1,2)로 flat시켜서 (42,4)로 변환
objp[:,:2] = np.mgrid[0:6,0:5].T.reshape(-1,2)*30
print(objp)

# 이미지로부터의 Object point와 Image points를 저장하기 위한 배열
objpoints = []
imgpoints = []

'''Find Corners'''
# Load Checkboard Images
images = glob.glob('./*.jpg')

for fname in images :
    cv.namedWindow(fname,cv.WINDOW_NORMAL)
    print(fname)
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # to gray
    
    
    # 체스판의 코너들 찾기
    ret, corners = cv.findChessboardCorners(gray, (6,5), None)
    print(ret)
    
    # 찾았으면, Object points, Image points 추가하기
    if ret == True:
        cv.imshow(fname,img)
        cv.resizeWindow(fname,500,500)
        cv.waitKey(1000)
        
        objpoints.append(objp)
        
        corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)
        
        # Draw Corners
        img = cv.drawChessboardCorners(img, (6,5), corners2, ret)
        cv.imwrite(fname+"new.jpg", img)
        cv.imshow(fname,img)
        cv.waitKey(2000)
cv.destroyAllWindows()

'''Calibration'''
ret, mtx, dist ,rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
print('ret :', ret)
print('mtx : ', mtx)
print('dist : ',dist)
print('rvecs : ',rvecs)
print('tvecs : ',tvecs)


img = cv.imread('./3.jpg')
h,w = img.shape[:2]
newcameraMtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

dst = cv.undistort(img,mtx,dist,None, newcameraMtx)

x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibRes.jpg', dst)


mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i],tvecs[i],mtx,dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
    
print("total error: {}".format(mean_error/len(objpoints)))


print(cv.calibrationMatrixValues(mtx,(w,h),1.4,1.4))
