import cv2
from cv2 import aruco
from ImageRegistrator import ImageRegistrator
import OpenCVHelper

if __name__ == "__main__":
    registrator = ImageRegistrator(cv2.imread('../data/im_ref.jpg'), aruco.Dictionary_get(aruco.DICT_4X4_50))
    for i in range(5):
        filename = '../data/im_temp_' + str(i + 1).zfill(2) +  '.jpg'
        im = cv2.imread(filename)
        im_reg = registrator.registerImage(im)
        im_contours = OpenCVHelper.drawContours(im_reg)
        OpenCVHelper.show(im_contours)