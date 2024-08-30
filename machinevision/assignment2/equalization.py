import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# Apply the histogram equalization to grayscale images with contrast issues. Draw the 
# histogram of the input and output images. Calculate the mean and standard deviation of 
# the input and output images. 

def histogramEqualization(image_array):
    