import cv2
import numpy as np

# Read image.
img = cv2.imread('C:/Users/agent/PycharmProjects/ML-and-Vision/machinevision/assignment3/cell.jpg', cv2.IMREAD_COLOR)

# Convert to grayscale.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)
# Blur using 3 * 3 kernel.
# gray = cv2.morphologyEx(gray, cv2.MORPH_DILATE, np.ones((21, 21), np.uint8))

(thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
gray = cv2.morphologyEx(gray, cv2.MORPH_DILATE, np.ones((19, 19), np.uint8))
# gray = cv2.morphologyEx(gray, cv2.MORPH_ERODE, np.ones((9, 9), np.uint8))

gray = cv2.blur(gray, (25, 25))
# gray = cv2.morphologyEx(gray, cv2.MORPH_DILATE, np.ones((15, 15), np.uint8))
# gray_blurred =
cv2.imshow('blurred', gray)
cv2.waitKey(0)

# Apply Hough transform on the blurred image.
detected_circles = cv2.HoughCircles(gray,
				cv2.HOUGH_GRADIENT, 1, 80, param1 = 50,
			param2 = 26, minRadius = 1, maxRadius = 50)

radii = []

# Draw circles that are detected.
if detected_circles is not None:

	# Convert the circle parameters a, b and r to integers.
	detected_circles = np.uint16(np.around(detected_circles))

	for pt in detected_circles[0, :]:
		a, b, r = pt[0], pt[1], pt[2]

		# Draw the circumference of the circle.
		cv2.circle(img, (a, b), r, (0, 255, 0), 2)
		print(r)
		radii.append(r)

		# Draw a small circle (of radius 1) to show the center.
		cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
		# cv2.imshow("Detected Circle", img)
		# cv2.waitKey(0)

# radii.sort()
print(f'min: {min(radii)}')
# radii.sort(reverse=True)
print(f'max: {max(radii)}')
print(f'total: {len(radii)}')
cv2.imshow('Detected Circles', img)
cv2.waitKey(0)
cv2.imwrite("C:/Users/agent/PycharmProjects/ML-and-Vision/machinevision/assignment3/cellcircled.jpg", img)


# For the cell.jpg image, write a code to count the total number of cells, calculate the size 
# of each cell in pixels, and show the boundary of the biggest cell in an output image. In 
# your code, you might use any techniques covered in this class. Hint: Thresholding, 
# morphological filters, connected components, etc.
