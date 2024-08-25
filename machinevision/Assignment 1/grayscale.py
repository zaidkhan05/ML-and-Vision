from PIL import Image
import numpy as np

# Change the gray level of an 8-bit gray level image (that has initially 256 gray 
# levels) to 128, 64, and 32 gray level images, respectively. Save each image to 
# show the effect. 
def reduce_gray_levels(image, howmany):
    # Initialize a new array with the same shape as the original image
    finalizedImages = []
    # reduced_image = np.zeros_like(image_array)
    imageArray = np.array(image)
    for i in range(howmany):
        bits = 7 - i
        # reduced_image = np.zeros(image_array.shape, dtype=np.ubyte)
            # """Reduce the bit depth of an image."""
        # Calculate the number of levels for the reduced bit depth
        levels = 2 ** bits
        
        # Calculate the quantization step size
        quantization_step = 256 // levels
        
        # Apply quantization using NumPy operations
        quantized_image_np = (imageArray // quantization_step) * quantization_step

        # finalizedImages.append(Image.fromarray(quantized_image_np))

        # #1,2,3
        # factor = i
        # scale = 2
        # scale = 2 ** (factor+1)
        # #downscale the image values by 2,4,8 to get them from 256 to 128,64,32
        # reduced_image = image_array // scale


        # # print(scale)
        # Convert the numpy array to a PIL image
        finalImage = Image.fromarray(quantized_image_np)
        #append the image to the list of images
        finalizedImages.append(finalImage)
        # image_array
    return finalizedImages



    
    

# Load the original image
image_path = "Assignment 1/images/rose.jpg"
image = Image.open(image_path).convert("L")  # Ensure it's grayscale
howmany = 3
# Reduce to 128, 64, and 32 gray levels manually
reducedImages = reduce_gray_levels(image, howmany)
# reducedImages = quantize(imageArray)
#save the images
for i in range(howmany):
    reducedImages[i].save(f"Assignment 1/images/rose_{2**(7-i)}.jpg")
    print(2**(7-i))
