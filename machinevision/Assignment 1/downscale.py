import numpy as np
# import image
from PIL import Image as pillow



# Down-sample any color or grayscale image from a 1024x1024 pixel-sized image 
# (such as rose.jpg) to 512x512, 256x256, 128x128, respectively. Then, up-sample 
# the images generated before, back to 1024x1024 pixels. Save each image to show 
# the effect.

# Down-sample the image to 512x512, 256x256, and 128x128 pixels
def downSample(img):
    # Initialize a new array to hold the new images
    downSampledImages = []
    # Convert the image to a numpy array
    image = np.array(img)
    for i in range(3):
        #get the size of the image
        sideSize = image.shape[0]
        #get the new size of the image by dividing by 2(1024, 512, 256, 128)
        newSideSize = sideSize // 2
        #check for rgb or grayscale image and make the new image array depending on that
        if len(image.shape) == 3:
            # RGB image
            newImageArray = np.zeros((newSideSize, newSideSize, image.shape[2]), dtype=np.uint8)
        else:
            # Grayscale image
            newImageArray = np.zeros((newSideSize, newSideSize), dtype=np.uint8)
        #loop through the new image array and get the average of the 4 pixels in the original image
        for y in range(newSideSize):
            for x in range(newSideSize):
                #get the average of the 4 pixels in the original image(multiply each value by .1 to avoid overflow)
                newVal = ((image[y*2,x*2]*.1+image[y * 2, x * 2+1]*.1)/2 +(image[y*2+1,x*2]*.1+image[y*2+1,x*2+1]*.1)/2)/2
                #divide by .1 to get the actual value
                newVal = newVal/.1
                #cant use this because it will overflow as max value is 32 for uint8
                # newVal = np.uint8(((image[y*2,x*2]+image[y * 2, x * 2+1]) +(image[y*2+1,x*2]+image[y*2+1,x*2+1]))/4)
                #set the new value to the new image array
                newImageArray[y, x] = newVal
        #append the new image to the downsampled images
        downSampledImages.append(pillow.fromarray(newImageArray))
        #make the new image our input image for the next iteration
        image = newImageArray
    #return the downsampled images
    return downSampledImages
    

# Up-sample the images back to 1024x1024 pixels
def upSampleTo1024(img):
    # Initialize a new array to hold the new images
    upSampledImages = []
    for i in range(3):
        # Convert the image to a numpy
        image = np.array(img[i])
        #check for rgb or grayscale image and make the new image array depending on that
        if len(image.shape) == 3:
            # RGB image
            newImageArray = np.zeros((1024, 1024, image.shape[2]), dtype=np.uint8)
        else:
            # Grayscale image
            newImageArray = np.zeros((1024, 1024), dtype=np.uint8)
        #get the size of the image
        ogSize = image.shape[0]
        #use the size to get the scale factor
        scale = 1024 // ogSize
        #loop through the new image array and set the value of the pixel to the value of the pixel in the original image
        for y in range(1024):
            for x in range(1024):
                newImageArray[y, x] = image[y // scale, x // scale]
        #append the new image to the upsampled images
        upSampledImages.append(pillow.fromarray(newImageArray))
    #return the upsampled images
    return upSampledImages
#Make a 2x2 image with the original image and the upsampled downsampled images
#not required so I used in built functions but it is a nice way to see the difference
def fullComparisonOfFinalImages(upSampledImages, img):
    # Initialize a new array to hold all the images
    allImages = []
    # Append the original image to the array
    allImages.append(img)
    # Append the upsampled images to the array
    for i in range(3):
        allImages.append(upSampledImages[i])
    # Create a new image with a large enough to contain all images
    newImage = pillow.new("RGB", (1024 * 2, 1024 * 2))
    # Paste the images into the new image
    x_offset = 0
    for i in range(2):
        y_offset = 0
        for j in range(2):
            newImage.paste(allImages[i + j * 2], (x_offset, y_offset))
            y_offset += 1024
        x_offset += 1024
    # Save the new image
    newImage.save("Assignment 1/images/rosecomparison.jpg")


# Load the original image
img = pillow.open('Assignment 1/images/rose.jpg')
# Down-sample the image to 512x512, 256x256, and 128x128 pixels
downSampledImages = downSample(img)
# Save the down-sampled images
downSampledImages[0].save('Assignment 1/images/rose512x512.jpg')
downSampledImages[1].save('Assignment 1/images/rose256x256.jpg')
downSampledImages[2].save('Assignment 1/images/rose128x128.jpg')
# Up-sample the images back to 1024x1024 pixels
upSampledImages = upSampleTo1024(downSampledImages)
# Save the up-sampled images
upSampledImages[0].save('Assignment 1/images/rose512x512to1024x1024.jpg')
upSampledImages[1].save('Assignment 1/images/rose256x256to1024x1024.jpg')
upSampledImages[2].save('Assignment 1/images/rose128x128to1024x1024.jpg')
# Compare the original image with the up-sampled images
fullComparisonOfFinalImages(upSampledImages, img)