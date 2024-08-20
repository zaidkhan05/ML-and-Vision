from PIL import Image
import numpy as np

# Change the gray level of an 8-bit gray level image (that has initially 256 gray 
# levels) to 128, 64, and 32 gray level images, respectively. Save each image to 
# show the effect. 
def reduce_gray_levels(image_array, num_levels):
    # Initialize a new array with the same shape as the original image
    reduced_image = np.zeros_like(image_array)
    
    # Calculate the quantization factor
    factor = 256 // num_levels
    # Reduce gray levels by integer division
    reduced_image = image_array // factor
    # return reduced_image
    return reduced_image
    
    

# Load the original image
image_path = "Assignment 1/images/rose.jpg"
image = Image.open(image_path).convert("L")  # Ensure it's grayscale
imageArray = np.array(image)

# Reduce to 128, 64, and 32 gray levels manually
reduced_128 = reduce_gray_levels(imageArray, 128)
reduced_64 = reduce_gray_levels(imageArray, 64)
reduced_32 = reduce_gray_levels(imageArray, 32)

# Convert numpy arrays back to images and save
Image.fromarray(reduced_128).save("roseAs128.jpg")
Image.fromarray(reduced_64).save("roseAs64.jpg")
Image.fromarray(reduced_32).save("roseAs32.jpg")
