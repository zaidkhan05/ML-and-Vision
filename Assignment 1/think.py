from PIL import Image
import numpy as np

def downsample(image_array, new_size):
    if len(image_array.shape) == 3:
        # RGB image
        downsampled = np.zeros((new_size, new_size, image_array.shape[2]), dtype=np.uint8)
    else:
        # Grayscale image
        downsampled = np.zeros((new_size, new_size), dtype=np.uint8)
    
    scale = image_array.shape[0] // new_size
    
    for i in range(new_size):
        for j in range(new_size):
            block = image_array[i*scale:(i+1)*scale, j*scale:(j+1)*scale]
            downsampled[i, j] = np.mean(block)
    
    return downsampled

def upsample(image_array, original_size):
    if len(image_array.shape) == 3:
        # RGB image
        upsampled = np.zeros((original_size, original_size, image_array.shape[2]), dtype=np.uint8)
    else:
        # Grayscale image
        upsampled = np.zeros((original_size, original_size), dtype=np.uint8)
    
    scale = original_size // image_array.shape[0]
    print(scale)
    for i in range(original_size):
        for j in range(original_size):
            upsampled[i, j] = image_array[i // scale, j // scale]
    
    return upsampled

# Load the original image
image_path = "Assignment 1/rose.jpg"
original_image = Image.open(image_path)
original_array = np.array(original_image)

# Down-sample to 512x512, 256x256, and 128x128
downsampled_512 = downsample(original_array, 512)
downsampled_256 = downsample(original_array, 256)
downsampled_128 = downsample(original_array, 128)

# Up-sample back to 1024x1024
upsampled_512 = upsample(downsampled_512, 1024)
upsampled_256 = upsample(downsampled_256, 1024)
upsampled_128 = upsample(downsampled_128, 1024)

# Convert numpy arrays back to images and save
Image.fromarray(downsampled_512).save("downsampled_512.jpg")
Image.fromarray(downsampled_256).save("downsampled_256.jpg")
Image.fromarray(downsampled_128).save("downsampled_128.jpg")
Image.fromarray(upsampled_512).save("upsampled_512.jpg")
Image.fromarray(upsampled_256).save("upsampled_256.jpg")
Image.fromarray(upsampled_128).save("upsampled_128.jpg")
