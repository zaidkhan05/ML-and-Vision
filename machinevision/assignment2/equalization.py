import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the grayscale image
image = cv2.imread('machinevision/assignment1/images/frog.jpg', cv2.IMREAD_GRAYSCALE)

# Calculate the histogram of the input image
input_hist, bins = np.histogram(image.flatten(), 256, [0,256])

# Apply histogram equalization
equalized_image = cv2.equalizeHist(image)

# Calculate the histogram of the output image
output_hist, bins = np.histogram(equalized_image.flatten(), 256, [0,256])

# Calculate the mean and standard deviation of the input and output images
input_mean = np.mean(image)
input_std = np.std(image)
output_mean = np.mean(equalized_image)
output_std = np.std(equalized_image)

# Plot the histograms
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Input Image')

plt.subplot(2, 2, 2)
plt.imshow(equalized_image, cmap='gray')
plt.title('Equalized Image')

plt.subplot(2, 2, 3)
plt.plot(input_hist, color='black')
plt.title('Input Image Histogram')

plt.subplot(2, 2, 4)
plt.plot(output_hist, color='black')
plt.title('Equalized Image Histogram')

plt.tight_layout()
plt.savefig('machinevision/assignment2/images/equalization.png')
plt.show()

# Display the mean and standard deviation
print(f'Input Image: Mean = {input_mean:.2f}, Std = {input_std:.2f}')
print(f'Equalized Image: Mean = {output_mean:.2f}, Std = {output_std:.2f}')

















# import cv2
# import numpy as np
# from PIL import Image


# # Apply the histogram equalization to grayscale images with contrast issues. Draw the 
# # histogram of the input and output images. Calculate the mean and standard deviation of 
# # the input and output images. 

# def histogramEqualization(image_array):
#     # Initialize a new array with the same shape as the original image
#     reduced_image = np.zeros_like(image_array)
#     # Loop through for 7-1 bits
#     for i in range(image.size[0]):
#         for j in range(image.size[1]):
#             reduced_image[i][j] = (image_array[i][j] - np.min(image_array)) / (np.max(image_array) - np.min(image_array)) * 255

#     # Convert the array to an image
#     reduced_image = Image.fromarray(reduced_image)
        
#     return reduced_image

# # Load the original image
# image_path = "machinevision/assignment2/fourierspectrum.pgm"
# image = Image.open(image_path).convert("L")  # Ensure it's grayscale
# imageArray = np.array(image)
# finalImage = histogramEqualization(imageArray)
# #save the image
# finalImage.save("machinevision/assignment2/fourierspectrum_histogramEqualized.jpg")