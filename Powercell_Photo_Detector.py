import numpy as np
import cv2

def editImage(image, lower_values, upper_values):
    global mask_cnts, original_resize, mask
    resize_image = cv2.imread(image)
    LENGTH, WIDTH = 640, 480
    resize_image = cv2.resize(resize_image, (LENGTH, WIDTH))
    original_resize = resize_image
    resize_image = cv2.GaussianBlur(resize_image, (7, 7), None)
    resize_image = cv2.erode(resize_image, None, iterations = 1)
    resize_image = cv2.dilate(resize_image, None, iterations = 1)
    HSV = cv2.cvtColor(resize_image, cv2.COLOR_BGR2HSV)
    lower = np.array(lower_values)
    upper = np.array(upper_values)
    mask = cv2.inRange(HSV, lower, upper)
    cv2.bitwise_and(HSV, HSV, mask = mask)
    mask_cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(mask_cnts) == 2:
        mask_cnts = mask_cnts[0]
    
    else:
        mask_cnts[1]
    
    return mask


def isCircle(contours):
    i = 0

    for contour in contours:
        if i == 0:
            i = 1
            continue
        # Approximate shape
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        
        # finding center point of shape
        M = cv2.moments(contour)
        if M['m00'] != 0.0:
            x = int(M['m10']/M['m00'])
            y = int(M['m01']/M['m00'])
            
        # Check if circle
        if len(approx) >= 12:
            return True
    

def drawRectangle(contours, image):
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        aspect_ratio = float(w)/h
        area = float(w*h)
        if isCircle(contours) and  1.2 > aspect_ratio > .5 and 1000 > area > 120:
            cv2.rectangle(original_resize, (x, y), (x + w, y + h), (0,255,0), 2)
        
        


#Draw the rectangles
editImage("powercelltest.jpg", [22, 93, 0], [45, 255, 255])
drawRectangle(mask_cnts, original_resize)
            
cv2.imshow("masked", mask)
cv2.imshow("original image", original_resize)
cv2.waitKey()
