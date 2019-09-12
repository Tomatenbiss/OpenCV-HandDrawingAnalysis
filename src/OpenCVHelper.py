import cv2
import numpy as np
from sklearn.decomposition import PCA

def dilateImage(im, kernelsize=5, iterations=1):
    """ 
    Applies OpenCVs dilate() method on the image.
    """
    kernel = np.ones((kernelsize,kernelsize),np.uint8)
    return cv2.dilate(im, kernel, iterations=1)

def drawContours(im, color=(255,255,255), contours=None):
    """ 
    Draws a set of contours on the image. If no contours are passed they 
    are found using OpenCV's findContours() method.
    """
    res = np.zeros(im.shape)
    if contours == None:
        contours, hierarchy = getContours(im)
    if len(contours) > 0:
        im = cv2.drawContours(res, contours, -1, color, cv2.FILLED)
    return im

def fitLineCV(im):
    """
    Fits a line through the white pixels of the image using OpenCV's fitline() 
    method and returns a black image with the drawn line.
    """
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
    
def fitLinePCA(im):
    """
    Fits a line through the white pixels of the image using sklearn's PCA class 
    and returns a black image with the drawn line.
    """
    empty = np.zeros(im.shape)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    pts = np.argwhere(gray == 255)
    if len(pts) > 0:
        pts[:,[0,1]] = pts[:,[1,0]]
        pca = PCA(n_components=2)
        pca.fit(pts)
        
        vector1 = np.add(pca.mean_, np.multiply(pca.components_[0], np.sqrt(pca.explained_variance_[0] * 3)))
        vector2 = np.add(pca.mean_, np.multiply(pca.components_[0], -np.sqrt(pca.explained_variance_[0] * 3)))
        
        im = cv2.line(empty, (int(pca.mean_[0]), int(pca.mean_[1])), (int(vector1[0]), int(vector1[1])), (0,255,0), 5)
        im = cv2.line(im, (int(pca.mean_[0]), int(pca.mean_[1])), (int(vector2[0]), int(vector2[1])), (0,255,0), 5)
        # im = cv2.line(im, pca.mean_, vector2, (0,255,0), 5)
    return im

def getBinaryImage(im):
    """
    Returns the image converted to a binary image.
    """
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _,res = cv2.threshold(im,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return res

def getContours(im):
    """
    Returns the contours found in the BGR image using OpenCV's findContours method.
    """
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _,im = cv2.threshold(im,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return (contours, hierarchy)

def filterContoursByArea(contours, threshold):
    """
    Returns all contours with an area greater than the threshold.
    """
    res_contours = []
    for contour in contours:        
        if cv2.contourArea(contour) > threshold: 
            res_contours.append(contour)
    return res_contours

def show(im, im_size=(900, 900)):
    """
    Displays the image in a window of size 900 x 900.
    """
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', im_size[0], im_size[1])
    cv2.imshow('image', im)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()