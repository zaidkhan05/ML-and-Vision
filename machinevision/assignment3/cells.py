import cv2
import numpy as np


def circledetector(image):
    # image = image.HoughCircles()
    # Convert to grayscale. 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    
    # Blur using 3 * 3 kernel. 
    # gray_blurred = cv2.blur(gray, (9, 9)) 
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    transformed = cv2.morphologyEx(laplacian, cv2.MORPH_DILATE, np.ones((7,7), np.uint8))
    cv2.imshow("Blurred", transformed)
    cv2.imwrite(f'machinevision/assignment3/results/{whichImage}blurred.jpg', transformed)
    
    # Apply Hough transform on the blurred image. 
    detected_circles = cv2.HoughCircles(transformed,  
                    cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, 
                param2 = 30, minRadius = 1, maxRadius = 40) 
    
    # Draw circles that are detected. 
    if detected_circles is not None: 
    
        # Convert the circle parameters a, b and r to integers. 
        detected_circles = np.uint16(np.around(detected_circles)) 
    
        for pt in detected_circles[0, :]: 
            a, b, r = pt[0], pt[1], pt[2] 
    
            # Draw the circumference of the circle. 
            cv2.circle(image, (a, b), r, (0, 255, 0), 2) 
    
            # Draw a small circle (of radius 1) to show the center. 
            cv2.circle(image, (a, b), 1, (0, 0, 255), 3) 
            # cv2.imshow("Detected Circle", image) 
            # cv2.waitKey(0) 

        #save the final image
    cv2.imwrite(f'machinevision/assignment3/results/{whichImage}circles.jpg', image)
        

    
    return image




inputPath = 'machinevision/assignment3/'
whichImage = 'cell'
image = cv2.imread(f'{inputPath}{whichImage}.jpg')
images = []
images.append(image)


#save the images to file
# cv2.imwrite(f'machinevision/assignment3/results/{whichImage}morphologicalFiltered.jpg', circledetection)

# For the cell.jpg image, write a code to count the total number of cells, calculate the size 
# of each cell in pixels, and show the boundary of the biggest cell in an output image. In 
# your code, you might use any techniques covered in this class. Hint: Thresholding, 
# morphological filters, connected components, etc.
