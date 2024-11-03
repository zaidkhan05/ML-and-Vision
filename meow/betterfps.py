import cv2
import numpy as np
import pandas as pd

# Load the image
matrix = [[-3,-1, 0, 7],
          [-3, 1,-3, 8],
          [-3, 2,-3, 4],
          [-2, 2,-2, 1]]
matrix = np.array(matrix, np.uint8)

image = cv2.resize(matrix, (256, 256), interpolation=cv2.INTER_NEAREST   )
cv2.imwrite('meow/resize.png', image)