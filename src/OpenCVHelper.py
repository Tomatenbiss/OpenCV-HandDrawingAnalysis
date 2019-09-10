import cv2
import numpy as np

def dilateImage(im, kernelsize=5, iterations=1):
    kernel = np.ones((kernelsize,kernelsize),np.uint8)
    return cv2.dilate(im, kernel, iterations=1)

def drawContours(im, color=(255,255,255), contours=None):
    res = np.zeros(im.shape)
    if contours == None:
        contours, hierarchy = getContours(im)
    if len(contours) > 0:
        im = cv2.drawContours(res, contours, -1, color, cv2.FILLED)
    return im

def fitLineCV(im):
    empty = np.zeros(im.shape)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    rows,cols = gray.shape
    pts = np.argwhere(gray == 255)
    pts[:,[0,1]] = pts[:,[1,0]]
    if len(pts) > 0:
        [vx,vy,x,y] = cv2.fitLine(pts, cv2.DIST_L1, 0, 1, 0.01)
        lefty = int((-x*vy/vx) + y)
        righty = int(((cols-x)*vy/vx)+y)    
        im = cv2.line(empty,(cols-1,righty),(0,lefty),(0,255,0),5)
    return im
    

def getContours(im):
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _,th = cv2.threshold(imgray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return (contours, hierarchy)

def filterContoursByArea(contours, threshold):
    res_contours = []
    for contour in contours:        
        if cv2.contourArea(contour) > threshold: 
            res_contours.append(contour)
    return res_contours

def show(im, im_size=(900, 900)):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', im_size[0], im_size[1])
    cv2.imshow('image', im)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()