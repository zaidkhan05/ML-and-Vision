import cv2
import numpy as np


def morphologicalFilter(input):
    # outputImage = np.zeros(input.shape)
    # for i in range(1, input.shape[0]-1):
    #     for j in range(1, input.shape[1]-1):
    #         outputImage[i,j] = np.max(input[i-1:i+2, j-1:j+2])
    image = cv2.morphologyEx(input, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return image
    # return outputImage

def medianFilter(input):
    # outputImage = np.zeros(input.shape)
    # for i in range(1, input.shape[0]-1):
    #     for j in range(1, input.shape[1]-1):
    #         outputImage[i,j] = np.median(input[i-1:i+2, j-1:j+2])
    image = cv2.medianBlur(input, 5)
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
whichImage = 'fingerPrint_BW'
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