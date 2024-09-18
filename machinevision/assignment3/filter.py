import cv2
import numpy as np


def morphologicalFilter(image):
    kernel = [[0,0,1,0,0], [0,1,1,1,0], [1,1,1,1,1], [0,1,1,1,0], [0,0,1,0,0]]
    kernel = np.array(kernel, np.uint8)
    ###############################################################
    # code for output image b
    # image = cv2.morphologyEx(input, cv2.MORPH_ERODE, kernel)
    ###############################################################

    ###############################################################
    # code for output image d
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    ###############################################################


    return image

def medianFilter(input):
    image = cv2.medianBlur(input, 7)
    # image = cv2.Laplacian(input, cv2.CV_64F)
    return image
def saveImages(images, whichImage):
    #save the images to one image canvas
    images = [image, morphologicalFiltered, medianFiltered]
    outputImage = np.zeros((images[0].shape[0], images[0].shape[1]*3, 3), np.uint8)
    outputImage[:, 0:images[0].shape[1]] = images[0]
    outputImage[:, images[0].shape[1]:images[0].shape[1]*2] = images[1]
    outputImage[:, images[0].shape[1]*2:] = images[2]

    cv2.imwrite(f"machinevision/assignment3/comparisons/{whichImage}comparison.jpg", outputImage)

inputPath = 'machinevision/assignment3/'
whichImage = 'morphology'
image = cv2.imread(f'{inputPath}{whichImage}.png')
images = []
images.append(image)
morphologicalFiltered = morphologicalFilter(image)
images.append(morphologicalFiltered)

medianFiltered = medianFilter(image)
images.append(medianFiltered)

saveImages(images, whichImage)
#save the images to file
cv2.imwrite(f'machinevision/assignment3/results/{whichImage}morphologicalFiltered.jpg', morphologicalFiltered)
cv2.imwrite(f'machinevision/assignment3/results/{whichImage}medianFiltered.jpg', medianFiltered)

# Apply both morphological and median filters on the fingerprint image 
# (fingerprint_BW.png). Compare the result and comment under what condition, one filter 
# might perform better than the othe