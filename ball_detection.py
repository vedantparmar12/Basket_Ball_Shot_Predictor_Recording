import cv2

image = cv2.imread("Files/Ball.png")

#image = image[0:900, :]

image_copy=image.copy()
# Step 1: Covert the Image from BGR to HSV Color Space
imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Ball Detection
lower_range = (7, 153, 29)
upper_range = (162, 255, 255)

mask = cv2.inRange(imageHSV, lower_range, upper_range)

ball_detection = cv2.bitwise_and(image, image, mask=mask)

# Find the Contours

contours, hirearchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Draw the Contours

cv2.drawContours(image_copy, contours, -1, (0,255,0), 3)
image_copy = cv2.resize(image_copy, (0,0), None, 0.6, 0.6)
cv2.imshow("Drawing Contours", image_copy)
cv2.waitKey(0)