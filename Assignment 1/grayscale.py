from PIL import Image
import numpy as np

# Change the gray level of an 8-bit gray level image (that has initially 256 gray 
# levels) to 128, 64, and 32 gray level images, respectively. Save each image to 
# show the effect. 
def reduce_gray_levels(image_array):
    # Initialize a new array with the same shape as the original image
    finalizedImages = []
    reduced_image = np.zeros_like(image_array)
    for i in range(3):
        #1,2,3
        factor = i
        scale = 2
        for y in range(factor):
            scale = scale*2
        #downscale the image values by 2,4,8 to get them from 256 to 128,64,32
        reduced_image = image_array // scale

        # print(scale)
        # Convert the numpy array to a PIL image
        finalImage = Image.fromarray(reduced_image)
        #append the image to the list of images
        finalizedImages.append(finalImage)
    return finalizedImages
    
    

# Load the original image
image_path = "Assignment 1/images/rose.jpg"
image = Image.open(image_path).convert("L")  # Ensure it's grayscale
imageArray = np.array(image)

# Reduce to 128, 64, and 32 gray levels manually
reducedImages = reduce_gray_levels(imageArray)
#save the images
for i in range(3):
    reducedImages[i].save(f"Assignment 1/images/rose_{2**(7-i)}.jpg")
    # print(2**(7-i))
