import cv2
from cv2 import aruco
import numpy as np

class ImageRegistrator:
    """ Registers photographs of hand-drawn to figures to a reference image using OpenCV's ArUco Markers. """
    def __init__(self, im_ref, aruco_dict):
        self.im_ref = im_ref
        self.aruco_dict = aruco_dict
        self.kp_ref = self.__getKeypoints(im_ref)
        

    def registerImage(self, im_temp):
        kp_temp = self.__getKeypoints(im_temp)
        M = cv2.getPerspectiveTransform(kp_temp, self.kp_ref)
        return cv2.warpPerspective(im_temp, M, (self.im_ref.shape[1], self.im_ref.shape[0]))

    def __getCentroids(self, corners, ids):
        centroids = {}
        for i in range(len(corners)):
            corner = corners[i]
            id = ids[i]
            x,y = np.matrix.transpose(np.array(corner))
            x = np.mean(x)
            y = np.mean(y)
            centroids[id[0]] = [x,y]
        centroids_list = []
        for i in range(len(corners)):
            centroids_list.append(centroids[i])
        return np.array(centroids_list)

    def __getKeypoints(self, im):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        parameters = aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.aruco_dict, parameters=parameters)
        return self.__getCentroids(corners, ids)
