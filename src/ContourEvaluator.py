import cv2
import ImageFilter
import numpy as np
import OpenCVHelper
import os

class ContourEvaluator:
    def __init__(self, mask_dir):
        self.masks = self.__getMasksFromDir(mask_dir)

    def __getMasksFromDir(self, mask_dir):
        # mask_dir = os.path.join(os.path.dirname(__file__), mask_dir)
        masks = []
        for i in range(16):
            masks.append(OpenCVHelper.getBinaryImage(cv2.imread(mask_dir + str(i + 1) + '.jpg')))
        return masks

    def __getTotalContourArea(self, im):
        contours,_ = OpenCVHelper.getContours(im)
        total_area = 0
        for contour in contours:
            total_area += cv2.contourArea(contour)
        return total_area



    def evaluateContours(self, im):
        im = im.astype('uint8')
        contours,_ = OpenCVHelper.getContours(im)
        im_contours = OpenCVHelper.drawContours(im)
        empty = np.zeros(im.shape[:2])
        total_points = 0
        for contour in contours:
            if cv2.contourArea(contour) > 0:
                im_contour = OpenCVHelper.drawContours(im, contours=np.array([contour]))
                points = 0
                for i in range(len(self.masks)):
                    im_mask = ImageFilter.filterByMask(im_contour, self.masks[i])
                    area = self.__getTotalContourArea(im_mask)
                    print('Contour Area: ' + str(area))
                    if area > 0:
                        points = (i + 1) * (i + 1)
                total_points += points
                print('Contour gets ' + str(points) + ' points.')
                
        return total_points

                
                


        

