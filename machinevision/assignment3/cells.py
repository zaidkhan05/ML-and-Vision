import cv2
import numpy as np
from PIL import Image


def circledetector(img):

    cv2.imwrite(f'machinevision/assignment3/results/{whichImage}circles.jpg', img)
    return img




image = cv2.imread('machinevision/assignment3/cell.jpg')
circledetection = circledetector(image)


# For the cell.jpg image, write a code to count the total number of cells, calculate the size 
# of each cell in pixels, and show the boundary of the biggest cell in an output image. In 
# your code, you might use any techniques covered in this class. Hint: Thresholding, 
# morphological filters, connected components, etc.
