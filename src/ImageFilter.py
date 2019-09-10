import cv2
import numpy as np

def filterHSVRed(im, h_range, s_values, v_values):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    lower = np.array([0, s_values[0], v_values[0]])
    upper = np.array([0 + h_range, s_values[1], v_values[1]])
    mask1 = cv2.inRange(im, lower, upper)
    
    lower = np.array([179 - h_range, s_values[0], v_values[0]])
    upper = np.array([179, s_values[1], v_values[1]])
    mask2 = cv2.inRange(im, lower, upper)

    mask_merged = cv2.bitwise_or(mask1, mask2)
    res = cv2.bitwise_and(im, im, mask=mask_merged)
    return cv2.cvtColor(res, cv2.COLOR_HSV2BGR)