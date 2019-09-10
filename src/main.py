import cv2
from cv2 import aruco
from ImageRegistrator import ImageRegistrator
import ImageFilter
import OpenCVHelper

if __name__ == "__main__":
    registrator = ImageRegistrator(cv2.imread('../data/im_ref.jpg'), aruco.Dictionary_get(aruco.DICT_4X4_50))
    for i in range(5):
        filename = '../data/im_temp_' + str(i + 1).zfill(2) +  '.jpg'
        im = cv2.imread(filename)
        im_reg = registrator.registerImage(im)
        im_filt = ImageFilter.filterHSVRed(im_reg, 5, (40, 255), (110, 170))
        im_filt = OpenCVHelper.dilateImage(im_filt)
        contours, hierarchy = OpenCVHelper.getContours(im_filt)
        contours = OpenCVHelper.filterContoursByArea(contours, 200)
        im_contours = OpenCVHelper.drawContours(im_filt, contours=contours)
        OpenCVHelper.show(im_contours)