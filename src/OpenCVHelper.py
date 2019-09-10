import cv2

def show(im, im_size=(900, 900)):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', im_size[0], im_size[1])
    cv2.imshow('image', im)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()

def drawContours(im, color=(0,255,0)):
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _,th = cv2.threshold(imgray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        im = cv2.drawContours(im.copy(), contours, -1, color, 3)
    return im
    
    