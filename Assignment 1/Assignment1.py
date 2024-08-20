import numpy as np
# import image
from PIL import Image as pillow

# Load the image
img = pillow.open('Assignment 1/rose.jpg')
# imgArray = np.array(img)

# Down-sample any color or grayscale image from a 1024x1024 pixel-sized image 
# (such as rose.jpg) to 512x512, 256x256, 128x128, respectively. Then, up-sample 
# the images generated before, back to 1024x1024 pixels. Save each image to show 
# the effect.

# Down-sample the image
# if len(image_array.shape) == 3:
#         # RGB image
#         downsampled = np.zeros((newSideSize, newSideSize, image_array.shape[2]), dtype=np.uint8)
#     else:
#         # Grayscale image
#         downsampled = np.zeros((newSideSize, newSideSize), dtype=np.uint8)
    
#     scale = image_array.shape[0] // newSideSize
    
#     for i in range(newSideSize):
#         for j in range(newSideSize):
#             block = image_array[i*scale:(i+1)*scale, j*scale:(j+1)*scale]
#             downsampled[i, j] = np.mean(block)
def downSample(img):
    downSampledImages = []
    image = np.array(img)
    for i in range(3):
        sideSize = image.shape[0]
        newSideSize = sideSize // 2
        # if len(image.shape) == 3:
        #     # RGB image
        #     newImageArray = np.zeros((newSideSize, newSideSize, image.shape[2]), dtype=np.uint8)
        # else:
        #     # Grayscale image
        #     newImageArray = np.zeros((newSideSize, newSideSize), dtype=np.uint8)
        
        # scale = image.shape[0] // newSideSize
        
        # for i in range(newSideSize):
        #     for j in range(newSideSize):
        #         block = image[i*scale:(i+1)*scale, j*scale:(j+1)*scale]
        #         newImageArray[i, j] = np.mean(block)
        #         if i == 24 and j == 24:
        #             print(block)

        newImageArray = np.zeros((newSideSize, newSideSize), dtype=np.uint8)
        for y in range(newSideSize):
            for x in range(newSideSize):
                newVal = ((image[y*2,x*2]*.1+image[y * 2, x * 2+1]*.1)/2 +(image[y*2+1,x*2]*.1+image[y*2+1,x*2+1]*.1)/2)/2
                if y == 25 and x == 25:
                    print(newVal)
                newImageArray[y, x] = newVal/.1
        # newImageArray = image[::2, ::2]
        downSampledImages.append(pillow.fromarray(newImageArray))
        image = newImageArray


    return downSampledImages
    

# Up-sample the images back to 1024x1024 pixels
def upSampleTo1024(img):
    upSampledImages = []
    for i in range(3):
        getImage = img[i]
        image = np.array(getImage)
        ogSize = image.shape[0]

        newImageArray = np.zeros((1024, 1024), dtype=np.uint8)
        # scale = 1024 // ogSize
        # for y in range(1024):
        #     for x in range(1024):
        #         newImageArray[y, x] = image[y // scale, x // scale]
        if ogSize == 512:
            for y in range(ogSize):
                for x in range(ogSize):
                    # newImageArray[y*2-1,x*2-1] = image[y, x]
                    # newImageArray[y * 2, x * 2] = image[y, x]
                    # newImageArray[y*2-1,x*2] = image[y, x]
                    # newImageArray[y*2,x*2-1] = image[y, x]
                    newImageArray[y*2-1: y*2+1, x*2-1: x*2+1] = image[y, x]
            upSampledImages.append(pillow.fromarray(newImageArray))
        elif ogSize == 256:
            for y in range(ogSize):
                for x in range(ogSize):
                    newImageArray[y*4-3,x*4-3] = image[y, x]
                    newImageArray[y*4-2,x*4-2] = image[y, x]
                    newImageArray[y*4-1,x*4-1] = image[y, x]
                    newImageArray[y*4,x*4] = image[y, x]
                    newImageArray[y*4-3,x*4] = image[y, x]
                    newImageArray[y*4,x*4-3] = image[y, x]
                    newImageArray[y*4-2,x*4] = image[y, x]
                    newImageArray[y*4,x*4-2] = image[y, x]
                    newImageArray[y*4-1,x*4] = image[y, x]
                    newImageArray[y*4,x*4-1] = image[y, x]
                    newImageArray[y*4-3,x*4-2] = image[y, x]
                    newImageArray[y*4-2,x*4-3] = image[y, x]
                    newImageArray[y*4-1,x*4-2] = image[y, x]
                    newImageArray[y*4-2,x*4-1] = image[y, x]
                    newImageArray[y*4-1,x*4-3] = image[y, x]

            upSampledImages.append(pillow.fromarray(newImageArray))
        elif ogSize == 128:
            for y in range(ogSize):
                for x in range(ogSize):
                    newImageArray[y*8-7,x*8-7] = image[y, x]
                    newImageArray[y*8-6,x*8-6] = image[y, x]
                    newImageArray[y*8-5,x*8-5] = image[y, x]
                    newImageArray[y*8-4,x*8-4] = image[y, x]
                    newImageArray[y*8-3,x*8-3] = image[y, x]
                    newImageArray[y*8-2,x*8-2] = image[y, x]
                    newImageArray[y*8-1,x*8-1] = image[y, x]
                    newImageArray[y*8,x*8] = image[y, x]
                    newImageArray[y*8-7,x*8] = image[y, x]
                    newImageArray[y*8,x*8-7] = image[y, x]
                    newImageArray[y*8-6,x*8] = image[y, x]
                    newImageArray[y*8,x*8-6] = image[y, x]
                    newImageArray[y*8-5,x*8] = image[y, x]
                    newImageArray[y*8,x*8-5] = image[y, x]
                    newImageArray[y*8-4,x*8] = image[y, x]
                    newImageArray[y*8,x*8-4] = image[y, x]
                    newImageArray[y*8-3,x*8] = image[y, x]
                    newImageArray[y*8,x*8-3] = image[y, x]
                    newImageArray[y*8-2,x*8] = image[y, x]
                    newImageArray[y*8,x*8-2] = image[y, x]
                    newImageArray[y*8-1,x*8] = image[y, x]
                    newImageArray[y*8,x*8-1] = image[y, x]
                    newImageArray[y*8-7,x*8-1] = image[y, x]
                    newImageArray[y*8-1,x*8-7] = image[y, x]
                    newImageArray[y*8-6,x*8-1] = image[y, x]
                    newImageArray[y*8-1,x*8-6] = image[y, x]
                    newImageArray[y*8-5,x*8-1] = image[y, x]
                    newImageArray[y*8-1,x*8-5] = image[y, x]
                    newImageArray[y*8-4,x*8-1] = image[y, x]
                    newImageArray[y*8-1,x*8-4] = image[y, x]
                    newImageArray[y*8-3,x*8-1] = image[y, x]
                    newImageArray[y*8-1,x*8-3] = image[y, x]
                    newImageArray[y*8-2,x*8-1] = image[y, x]
                    newImageArray[y*8-1,x*8-2] = image[y, x]
                    newImageArray[y*8-7,x*8-2] = image[y, x]
                    newImageArray[y*8-2,x*8-7] = image[y, x]
                    newImageArray[y*8-6,x*8-2] = image[y, x]
                    newImageArray[y*8-2,x*8-6] = image[y, x]
                    newImageArray[y*8-5,x*8-2] = image[y, x]
                    newImageArray[y*8-2,x*8-5] = image[y, x]
                    newImageArray[y*8-4,x*8-2] = image[y, x]
                    newImageArray[y*8-2,x*8-4] = image[y, x]
                    newImageArray[y*8-3,x*8-2] = image[y, x]
                    newImageArray[y*8-2,x*8-3] = image[y, x]
                    newImageArray[y*8-7,x*8-3] = image[y, x]
                    newImageArray[y*8-3,x*8-7] = image[y, x]
                    newImageArray[y*8-6,x*8-3] = image[y, x]
                    newImageArray[y*8-3,x*8-6] = image[y, x]
                    newImageArray[y*8-5,x*8-3] = image[y, x]
                    newImageArray[y*8-3,x*8-5] = image[y, x]
                    newImageArray[y*8-4,x*8-3] = image[y, x]
                    newImageArray[y*8-3,x*8-4] = image[y, x]
                    newImageArray[y*8-7,x*8-4] = image[y, x]
                    newImageArray[y*8-4,x*8-7] = image[y, x]
                    newImageArray[y*8-6,x*8-4] = image[y, x]
                    newImageArray[y*8-4,x*8-6] = image[y, x]
                    newImageArray[y*8-5,x*8-4] = image[y, x]
                    newImageArray[y*8-4,x*8-5] = image[y, x]
                    newImageArray[y*8-7,x*8-5] = image[y, x]
                    newImageArray[y*8-5,x*8-7] = image[y, x]
                    newImageArray[y*8-6,x*8-5] = image[y, x]
                    newImageArray[y*8-5,x*8-6] = image[y, x]
                    newImageArray[y*8-7,x*8-6] = image[y, x]
                    newImageArray[y*8-6,x*8-7] = image[y, x]

            upSampledImages.append(pillow.fromarray(newImageArray))
        # img = newImageArray

    return upSampledImages

# Change the gray level of an 8-bit gray level image (that has initially 256 gray 
# levels) to 128, 64, and 32 gray level images, respectively. Save each image to 
# show the effect. 


downSampledImages = downSample(img)
downSampledImages[0].save('Assignment 1/rose512x512.jpg')
downSampledImages[1].save('Assignment 1/rose256x256.jpg')
downSampledImages[2].save('Assignment 1/rose128x128.jpg')
upSampledImages = upSampleTo1024(downSampledImages)
upSampledImages[0].save('Assignment 1/rose512x512to1024x1024.jpg')
upSampledImages[1].save('Assignment 1/rose256x256to1024x1024.jpg')
upSampledImages[2].save('Assignment 1/rose128x128to1024x1024.jpg')