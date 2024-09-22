import cv2
import numpy as np
#Code referenced from:
#https://www.geeksforgeeks.org/circle-detection-using-opencv-python/

# Read image.
img = cv2.imread('cell.jpg', cv2.IMREAD_COLOR)

# Convert to grayscale.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)
(thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
gray = cv2.morphologyEx(gray, cv2.MORPH_DILATE, np.ones((19, 19), np.uint8))
gray = cv2.blur(gray, (25, 25))
cv2.imshow('blurred', gray)
cv2.waitKey(0)

# Apply Hough transform on the blurred image.
detected_circles = cv2.HoughCircles(gray,
				cv2.HOUGH_GRADIENT, 1, 80, param1 = 50,
			param2 = 26, minRadius = 1, maxRadius = 50)
#for figuring out if the thing works
radii = []

# Draw circles that are detected.
if detected_circles is not None:

	# Convert the circle parameters a, b and r to integers.
	detected_circles = np.uint16(np.around(detected_circles))

	for pt in detected_circles[0, :]:
		a, b, r = pt[0], pt[1], pt[2]

		# Draw the circumference of the circle.
		cv2.circle(img, (a, b), r, (0, 255, 0), 2)
		# print(r)
		radii.append(r)

		# Draw a small circle (of radius 1) to show the center.
		cv2.circle(img, (a, b), 1, (0, 0, 255), 3)

print(detected_circles)
#smallest radius
print(f'min: {min(radii)}')
#largest radius
print(f'max: {max(radii)}')
#how many circles
print(f'total: {len(radii)}')
cv2.imshow('Detected Circles', img)
cv2.waitKey(0)
cv2.imwrite("cellcircled.jpg", img)


