import cv2
from cv2 import aruco
from ImageRegistrator import ImageRegistrator
import OpenCVHelper

if __name__ == "__main__":
    registrator = ImageRegistrator(cv2.imread('../data/im_ref.jpg'), aruco.Dictionary_get(aruco.DICT_4X4_50))
    OpenCVHelper.show(registrator.registerImage(cv2.imread('../data/im_temp_01.jpg')))