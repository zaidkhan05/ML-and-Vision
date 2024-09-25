import cv2
import numpy as np
import matplotlib.pyplot as plt

def equalizeImage(image):
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
    plt.savefig('machinevision/assignment2/comparisons/roseequalization.png')
    plt.show()

    # Display the mean and standard deviation
    print(f'Input Image: Mean = {input_mean:.2f}, Std = {input_std:.2f}')
    print(f'Equalized Image: Mean = {output_mean:.2f}, Std = {output_std:.2f}')

    return equalized_image



image = cv2.imread('machinevision/assignment1/images/rose.jpg', cv2.IMREAD_GRAYSCALE)
equalized_image = equalizeImage(image)
cv2.imwrite('machinevision/assignment2/images/roseequalized.jpg', equalized_image)


