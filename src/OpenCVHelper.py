import cv2

def show(im, im_size=(900, 900)):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', im_size[0], im_size[1])
    cv2.imshow('image', im)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()