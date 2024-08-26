from PIL import Image
import numpy as np

# Change the gray level of an 8-bit gray level image (that has initially 256 gray 
# levels) to 128, 64, and 32 gray level images, respectively. Save each image to 
# show the effect. 
def quantizer(image_array):
    # Initialize a new array with the same shape as the original image
    finalizedImages = []
    reduced_image = np.zeros_like(image_array)
    # Loop through for 7-1 bits
    for i in range(7):
        # Calculate the number of levels and the quantization
        bits = 7 - i
        levels = 2 ** bits
        quantization_step = 256 // levels
        print(quantization_step)
        # integer division of the matrix by the quantization step, then multiplying by the quantization step to brighten the image
        quantized_image_np = (image_array // quantization_step) * quantization_step
        reduced_image = quantized_image_np

        # Convert the array to an image
        reduced_image = Image.fromarray(reduced_image)
        finalizedImages.append(reduced_image)
        
    return finalizedImages

#Make a 4x2 image with the original image and the upsampled downsampled images
#not required so I used in built functions but it is a nice way to see the difference
def fullComparisonOfFinalImages(reducedImages, img):
    # Initialize a new array to hold all the images
    allImages = []
    # Append the original image to the array
    allImages.append(img)
    # Append the upsampled images to the array
    for i in range(reducedImages.__len__()):
        allImages.append(reducedImages[i])
    # Create a new image large enough to contain all images
    x, y = img.size
    newImage = Image.new("RGB", (x * 4, y*2))
    # Paste the images into the new image
    x_offset = 0
    k=0
    for i in range(2):
        y_offset = 0
        for j in range(4):
            print(k)
            newImage.paste(allImages[k], (y_offset, x_offset))
            print(x_offset/x, y_offset/y)
            k += 1
            y_offset += y
        x_offset += x
    # Save the new image 
    newImage.save("machinevision/assignment1/comparisons/rosequantizecomparison.jpg")
    
    

# Load the original image
image_path = "machinevision/assignment1/images/rose.jpg"
image = Image.open(image_path).convert("L")  # Ensure it's grayscale
imageArray = np.array(image)

# Reduce to 7-1 gray levels manually
reducedImages = quantizer(imageArray)
#save the images
for i in range(reducedImages.__len__()):
    reducedImages[i].save(f"machinevision/assignment1/images/rosequantizeas{2**(7-i)}.jpg")
    # print(2**(7-i))
fullComparisonOfFinalImages(reducedImages, image)
