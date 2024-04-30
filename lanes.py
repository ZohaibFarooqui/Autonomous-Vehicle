# import cv2
# import numpy as np
# #import matplotlib.pyplot as plt
# def Region_of_interest(image):
#     height = image.shape[0]
#     polygons = np.array([
#         [(200,height),(1100,height),(550,250)]
#     ])
#     mask = np.zeros_like(image)  #To get same pixels(rows and columns) as our image
#     cv2.fillPoly(mask,polygons,255)
#     masked_image = cv2.bitwise_and(image,mask)#cropped the image by only doing bitwise and
#     return masked_image
#
# def Canny(image):
#     # makes image gray
#     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
#     canny = cv2.Canny(blur, 50, 150)
#     return canny
# def Display_lines(image,lines):
#     line_image = np.zeros_like(image)
#     if lines is not None:
#         for x1,y1,x2,y2 in lines:
#             cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
#     return line_image
#
# def make_coordinates(image,line_parameters):
#     slope,intercept = line_parameters
#     y1 = image.shape[0]
#     y2 = int(y1*(3/5))
#     x1 = int((y1 - intercept)/slope)
#     x2 = int((y2 - intercept)/slope)
#     return np.array([x1,y1,x2,y2])
# def Average_slope_intercept(image, lines):
#     left_fit = []
#     right_fit = []
#     for line in lines:
#         x1, y1, x2, y2 = line.reshape(4)
#         # We get the parameters m and b of a one-order line to fit these 2 points
#         parameters = np.polyfit((x1,x2), (y1,y2), 1)
#         slope = parameters[0]
#         intercept = parameters[1]
#         # y is reversed => left line has negative slope
#         if slope<0:
#             left_fit.append((slope,intercept))
#         else:
#             right_fit.append((slope,intercept))
#     # Then we get the average slope
#     left_fit_average = np.average(left_fit, axis = 0)
#     right_fit_average = np.average(right_fit, axis = 0)
#     left_fit_average = np.average(left_fit,axis=0)
#     right_fit_average = np.average(right_fit, axis=0)
#     left_line = make_coordinates(image,left_fit_average)
#     right_line = make_coordinates(image,right_fit_average)
#     # print(left_fit_average)
#     # print(right_fit_average)
#     return np.array([left_line,right_line])
#
#
#
# # image = cv2.imread('test_image.jpg')
# # lane_image = np.copy(image)
# # canny_image = Canny(lane_image)
# # region_of_interest = Region_of_interest(canny_image)
# # cropped_image = Region_of_interest(canny_image)
# # lines = cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40, maxLineGap=5) #1radian
# # averaged_lines = Average_slope_intercept(lane_image,lines)
# # line_image = Display_lines(lane_image,averaged_lines)    #blue lanes
# # combo_image = cv2.addWeighted(lane_image,0.8,line_image,1,1)
# # cv2.imshow('result',combo_image)
# # #plt.imshow(canny)
# # cv2.waitKey(0)
# #plt.show()
#
#
# cap = cv2.VideoCapture("test2.mp4")
# while(cap.isOpened()):
#     _, frame = cap.read() # returns a boolean we are not interested now, and the frame
#     canny_image = Canny(frame)
#     cropped_image = Region_of_interest(canny_image)
#     lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
#     averaged_lines = Average_slope_intercept(frame, lines)
#     line_image = Display_lines(frame, averaged_lines)
#     combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
#     cv2.imshow('result', combo_image)
#
#     # Displays the image for a specific amount of time
#     # With 1 it will wait 1ms in between the frames. Ιf we press q, then break
#     if cv2.waitKey(1) == ord('q'):
#         break
# cap.release()

import cv2
import numpy as np


def region_of_interest(image):
    height, width = image.shape
    region_mask = np.zeros_like(image)

    vertices = np.array([[(200, height), (1100, height), (550, 250)]], dtype=np.int32)
    cv2.fillPoly(region_mask, vertices, 255)

    masked_image = cv2.bitwise_and(image, region_mask)
    return masked_image


def canny(image):
    if image is None or image.size == 0:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0) if left_fit else None
    right_fit_average = np.average(right_fit, axis=0) if right_fit else None

    return left_fit_average, right_fit_average


def draw_lines(image, lines):
    line_image = np.zeros_like(image)

    if lines[0] is not None:
        left_line = make_coordinates(image, lines[0])
        cv2.line(line_image, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (255, 0, 0), 10)

    if lines[1] is not None:
        right_line = make_coordinates(image, lines[1])
        cv2.line(line_image, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (255, 0, 0), 10)

    return line_image


cap = cv2.VideoCapture("test2.mp4")  # Use 0 for default webcam

while cap.isOpened():
    print("Before reading frame")
    _, frame = cap.read()
    print("After reading frame")

    if frame is None:
        print("Frame is None. Exiting.")
        break

    canny_image = canny(frame)
    if canny_image is None:
        print("Error in frame processing.")
        continue

    region_of_interest_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(region_of_interest_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)

    left_fit, right_fit = average_slope_intercept(frame, lines)
    line_image = draw_lines(frame, (left_fit, right_fit))
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    cv2.imshow('result', combo_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
