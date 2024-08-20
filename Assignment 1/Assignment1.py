import numpy as np
import image
from PIL import Image as pillow

# Load the image
img = pillow.open('Assignment 1/rose.jpg')
imgArray = np.array(img)

# Down-sample any color or grayscale image from a 1024x1024 pixel-sized image 
# (such as rose.jpg) to 512x512, 256x256, 128x128, respectively. Then, up-sample 
# the images generated before, back to 1024x1024 pixels. Save each image to show 
# the effect.

# Down-sample the image
def downSample(img):
    downSampledImages = []
    for i in range(3):
        sideSize = img.shape[1]
        # print (sideSize)
        newSideSize = sideSize // 2
        newImageArray = np.zeros((newSideSize, newSideSize), dtype=np.uint8)
        print(newImageArray.shape[1])
        for y in range(newSideSize):
            for x in range(newSideSize):
                newImageArray[y, x] = img[y * 2, x * 2]

        downSampledImages.append(pillow.fromarray(newImageArray))
        img = newImageArray

    return downSampledImages

# Up-sample the images back to 1024x1024 pixels
def upSampleTo1024(img):
    upSampledImages = []
    for i in range(3):
        image = np.array(img[i])

        # sideSize = img.shape[1]
        # newSideSize = sideSize * 2
        scale = 1024 // image.shape[0]

        newImageArray = np.zeros((1024, 1024), dtype=np.uint8)
        for y in range(1024):
            for x in range(1024):
                newImageArray[y, x] = image[y // scale, x // scale]
        upSampledImages.append(pillow.fromarray(newImageArray))
        img = newImageArray

    return upSampledImages

# Change the gray level of an 8-bit gray level image (that has initially 256 gray 
# levels) to 128, 64, and 32 gray level images, respectively. Save each image to 
# show the effect. 


downSampledImages = downSample(imgArray)
downSampledImages[0].save('Assignment 1/rose512x512.jpg')
downSampledImages[1].save('Assignment 1/rose256x256.jpg')
downSampledImages[2].save('Assignment 1/rose128x128.jpg')
upSampledImages = upSampleTo1024(downSampledImages)
upSampledImages[0].save('Assignment 1/rose512x512to1024x1024.jpg')
upSampledImages[1].save('Assignment 1/rose256x256to1024x1024.jpg')
upSampledImages[2].save('Assignment 1/rose128x128to1024x1024.jpg')