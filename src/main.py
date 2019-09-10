import cv2
from cv2 import aruco
from ImageRegistrator import ImageRegistrator

def show(im):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 900, 900)
    cv2.imshow('image', im)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    registrator = ImageRegistrator(cv2.imread('../data/im_ref.jpg'), aruco.Dictionary_get(aruco.DICT_4X4_50))
    show(registrator.registerImage(cv2.imread('../data/im_temp_01.jpg')))