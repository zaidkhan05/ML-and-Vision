import numpy as np
from PIL import Image

# The general form of the log transformation function is s=T(r)=c*log(1+r), where 's' and 'r' 
# are the output and input pixel values, and 'c' is the scaling constant. Implement log 
# transformation and power law transformation. Use the image (fourierspectrum.pgm) 
# to test the algorithms. Comment on the similarity/difference between log transformation 
# and power-law transformation.



def logTransformation(image_array, c):
    # Initialize a new array with the same shape as the original image
    finalizedImages = []
    reduced_image = np.zeros_like(image_array)
    # Loop through for 7-1 bits
        
    # integer division of the matrix by the quantization step, then multiplying by the quantization step to brighten the image
    quantized_image_np = np.zeros_like(image_array)
    for i in range(image.size[0]):
        for j in range(image.size[1]):
            quantized_image_np[i][j] = (c * np.log(1 + image_array[i][j]))


    reduced_image = quantized_image_np

    # Convert the array to an image
    reduced_image = Image.fromarray(reduced_image)
    finalizedImages.append(reduced_image)
        
    return finalizedImages

def powerLawTransformation(image_array, c, gamma):
    # Initialize a new array with the same shape as the original image
    finalizedImages = []
    reduced_image = np.zeros_like(image_array)
    # Loop through for 7-1 bits
        
    # integer division of the matrix by the quantization step, then multiplying by the quantization step to brighten the image
    quantized_image_np = np.zeros_like(image_array)
    for i in range(image.size[0]):
        for j in range(image.size[1]):
            quantized_image_np[i][j] = (c * (image_array[i][j])**gamma)


    reduced_image = quantized_image_np

    # Convert the array to an image
    reduced_image = Image.fromarray(reduced_image)
    finalizedImages.append(reduced_image)
        
    return finalizedImages

# #Make a 4x2 image with the original image and the upsampled downsampled images
# #not required so I used in built functions but it is a nice way to see the difference
# def fullComparisonOfFinalImages(reducedImages, img):
#     # Initialize a new array to hold all the images
#     allImages = []
#     # Append the original image to the array
#     allImages.append(img)
#     # Append the upsampled images to the array
#     for i in range(reducedImages.__len__()):
#         allImages.append(reducedImages[i])
#     # Create a new image large enough to contain all images
#     x, y = img.size
#     newImage = Image.new("RGB", (x * 4, y*2))
#     # Paste the images into the new image
#     x_offset = 0
#     k=0
#     for i in range(2):
#         y_offset = 0
#         for j in range(4):
#             print(k)
#             newImage.paste(allImages[k], (y_offset, x_offset))
#             print(x_offset/x, y_offset/y)
#             k += 1
#             y_offset += y
#         x_offset += x
#     # Save the new image 
#     newImage.save("machinevision/assignment1/comparisons/rosequantizecomparison.jpg")
    
    

# Load the original image
image_path = "machinevision/assignment2/fourierspectrum.pgm"
image = Image.open(image_path).convert("L")  # Ensure it's grayscale
imageArray = np.array(image)

# Reduce to 7-1 bits
logTransform = logTransformation(imageArray, 50)
powerLawTransform = powerLawTransformation(imageArray, 50, 0.9)
#save the images
logTransform[0].save("machinevision/assignment2/fourierspectrum_logTransformed.jpg")
powerLawTransform[0].save("machinevision/assignment2/fourierspectrum_powerLawTransformed.jpg")
    # print(2**(7-i))
# fullComparisonOfFinalImages(reducedImages, image)
