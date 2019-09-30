import cv2
import OpenCVHelper
import os

class ContourEvaluator:
    def __init__(self, mask_dir):
        self.masks = self.__getMasksFromDir(mask_dir)

    def __getMasksFromDir(self, mask_dir):
        # mask_dir = os.path.join(os.path.dirname(__file__), mask_dir)
        masks = []
        for i in range(16):
            masks.append(cv2.imread(mask_dir + str(i + 1) + '.png'))
        return masks

    def evaluateContours(self, im):
        OpenCVHelper.show(im)

