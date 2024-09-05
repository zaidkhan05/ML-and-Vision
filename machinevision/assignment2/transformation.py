import numpy as np
from PIL import Image

# The general form of the log transformation function is s=T(r)=c*log(1+r), where 's' and 'r' 
# are the output and input pixel values, and 'c' is the scaling constant. Implement log 
# transformation and power law transformation. Use the image (fourierspectrum.pgm) 
# to test the algorithms. Comment on the similarity/difference between log transformation 
# and power-law transformation.

#log transformation function
def logTransformation(image_array, c):
    #initialize a new array for the image
    transformedImage = np.zeros_like(image_array)
    #apply the transformation
    for i in range(image_array.shape[0]):
        for j in range(image_array.shape[1]):
            #s=T(r)=c*log(1+r)
            transformedImage[i][j] = (c * np.log(1 + image_array[i][j]))
    #convert the transformed array into an image
    transformedImage = Image.fromarray(transformedImage)
    return transformedImage

#power law transformation function
def powerLawTransformation(image_array, c, gamma):
    #initialize a new array for the image
    transformedImage = np.zeros_like(image_array)
    #apply the transformation
    for i in range(image_array.shape[0]):
        for j in range(image_array.shape[1]):
            #s=T(r)=c*r^gamma
            transformedImage[i][j] = (c * (image_array[i][j])**gamma)
    #convert the transformed array into an image
    transformedImage = Image.fromarray(transformedImage)
    return transformedImage

#save the images with the original image
def fullComparisonOfFinalImages(images):
    # Create a new image with a large enough to contain all images
    newImage = Image.new("RGB", (images[0].size[0] * 3, images[0].size[1]))
    # Paste the images into the new image
    x_offset = 0
    for i in range(3):
        y_offset = 0
        for j in range(1):
            newImage.paste(images[i + j * 2], (x_offset, y_offset))
            y_offset += images[0].size[1]
        x_offset += images[0].size[0]
    # Save the new image
    return newImage


# Load the original image
image_path = "machinevision/assignment2/fourierspectrum.pgm"
image = Image.open(image_path).convert("L")  # Ensure it's grayscale
imageArray = np.array(image)
c=50
gamma=0.8
#do the transformations
logTransform = logTransformation(imageArray, c)
powerLawTransform = powerLawTransformation(imageArray, c, gamma)
images = [image, logTransform, powerLawTransform]
#save a comparison of the images
comparisons = fullComparisonOfFinalImages(images)
#save the images
comparisons.save("machinevision/assignment2/comparisons/fourierspectrumtransformationcomparison.jpg")
logTransform.save("machinevision/assignment2/images/fourierspectrumlogTransformed.jpg")
powerLawTransform.save("machinevision/assignment2/images/fourierspectrumpowerLawTransformed.jpg")

