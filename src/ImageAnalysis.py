import cv2
import OpenCVHelper

def getPixelSizeA4(im):
    return getPixelSize(im, 192, 272)

def getPixelSize(im, width_mm, height_mm):
    pixel_width = width_mm / im.shape[1]
    pixel_height = height_mm / im.shape[0]
    return (pixel_height + pixel_width) / 2

def getTotalContourLength(im):
    contours, _ = OpenCVHelper.getContours(im.astype('uint8'))
    pixel_size = getPixelSizeA4(im)
    print(pixel_size)
    total_length = 0
    for contour in contours:
        arc_len = cv2.arcLength(contour, False)
        estimated_line_len = arc_len / 2 * pixel_size
        total_length += estimated_line_len
    return total_length
    