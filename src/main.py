import ContourEvaluator
import cv2
from cv2 import aruco
import ImageAnalysis
from ImageRegistrator import ImageRegistrator
from ImageTiler import ImageTiler
import ImageFilter
import OpenCVHelper


if __name__ == "__main__":
    registrator = ImageRegistrator(cv2.imread('../data/im_ref.jpg'), aruco.Dictionary_get(aruco.DICT_4X4_50))
    # # for i in range(1):
    filename = '../data/im_temp_10.jpg'
    im = cv2.imread(filename)
    im_reg = registrator.registerImage(im)
    # OpenCVHelper.show(im_reg)
    # cv2.imwrite('../data/im_reg_' + str(i + 1).zfill(2) + '.jpg', im_reg)
    im_filt = ImageFilter.filterHSVRed(im_reg, 10, (40, 255), (110, 170))
    im_filt = OpenCVHelper.dilateImage(im_filt)
    contours, hierarchy = OpenCVHelper.getContours(im_filt)
    contours = OpenCVHelper.filterContoursByArea(contours, 200)
    im_contours = OpenCVHelper.drawContours(im_filt.copy(), contours=contours)
    
    im_mask = OpenCVHelper.getBinaryImage(cv2.imread(('../data/mask/teddy-10-outer-mask.jpg')))
    im_mask = OpenCVHelper.getBinaryImage(cv2.imread(('../data/mask/AreaMask.jpg')))
    
    im_final = ImageFilter.filterByMask(im_contours, im_mask)
    eval = ContourEvaluator.ContourEvaluator('../data/mask/distance/')
    points = eval.evaluateContours(im_final)
    total_contour_length = ImageAnalysis.getTotalContourLength(im_final)
    print('Length: ' + str(total_contour_length))
    print('Distance Error: ' + str(points))
    print('Total: ' + str(int(total_contour_length / 5) + points))
    
    im_mask_inverted = OpenCVHelper.invert(im_mask)
    im_combined = OpenCVHelper.combineImages(im_final, im_mask_inverted)
    OpenCVHelper.show(im_combined)
