import cv2
import numpy as np
#import matplotlib.pyplot as plt

#grayscale, cleaning(blur)
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)# (optional) reducing the noise
    #We need a gradient image to simplify
    canny = cv2.Canny(blur, 50,150)
    return canny

def display_lines(image,lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(line_image, (x1,y1), (x2,y2), (0,255,0), 10)
    return line_image

#finding our region of interesty by a figure
def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[(200,height),(1100,height),(550,250)]])
    mask = np.zeros_like(image)
    #fillPoly needs more than one polygon so we make a list
    cv2.fillPoly(mask,polygons,255)
    masked_image = cv2.bitwise_and(image,mask)#and po bitah img
    return masked_image

#reading the image
image = cv2.imread("Photos/test_image.jpg")
#matrix of the image
lane_image = np.copy(image)
canny_immage = canny(lane_image)
cropped_image =  region_of_interest(canny_immage)
lines = cv2.HoughLinesP(cropped_image,2,np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5, )
line_image = display_lines(lane_image,lines)
combo_image = cv2.addWeighted(lane_image,0.8,line_image,1, 1)
#showing the image (window title, image)
cv2.imshow("result", combo_image)
#displaying image unitl key press
cv2.waitKey(0)

#we need pyplot to see the the numerated diagram of image for the base of triangle
#plt.imshow(canny)
#plt.show()
