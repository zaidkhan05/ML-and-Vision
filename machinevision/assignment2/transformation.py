import numpy as np
from PIL import Image

# The general form of the log transformation function is s=T(r)=c*log(1+r), where 's' and 'r' 
# are the output and input pixel values, and 'c' is the scaling constant. Implement log 
# transformation and power law transformation. Use the image (fourierspectrum.pgm) 
# to test the algorithms. Comment on the similarity/difference between log transformation 
# and power-law transformation.

def logTransformation(image_array, c):
    transformedImage = np.zeros_like(image_array)
    for i in range(image_array.shape[0]):
        for j in range(image_array.shape[1]):
            transformedImage[i][j] = (c * np.log(1 + image_array[i][j]))
    transformedImage = Image.fromarray(transformedImage)
    return transformedImage

def powerLawTransformation(image_array, c, gamma):
    transformedImage = np.zeros_like(image_array)
    for i in range(image_array.shape[0]):
        for j in range(image_array.shape[1]):
            transformedImage[i][j] = (c * (image_array[i][j])**gamma)
    transformedImage = Image.fromarray(transformedImage)
    return transformedImage


# Load the original image
image_path = "machinevision/assignment2/banker.jpg"
image = Image.open(image_path).convert("L")  # Ensure it's grayscale
imageArray = np.array(image)

#do the transformations
logTransform = logTransformation(imageArray, 50)
powerLawTransform = powerLawTransformation(imageArray, 25, 0.6)
#save the images
logTransform.save("machinevision/assignment2/images/bankerfourierspectrum_logTransformed.jpg")
powerLawTransform.save("machinevision/assignment2/images/bankerfourierspectrum_powerLawTransformed.jpg")

