import cv2
import numpy as np
from pdf2image import convert_from_path
import os
import matplotlib.pyplot as plt
import pandas as pd
import pytesseract

img_path = 'data/image/enhanced.png'
rot_img_path = 'data/image/rotated_img.png'

# def draw_contours_around_text(image_path):
#     # Read the image
#     image = cv2.imread(image_path)

#     # Convert the image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Apply a slight Gaussian blur to the image
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)

#     # Threshold the image
#     _, thresholded = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY_INV)

#     # Find contours in the thresholded image
#     contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Iterate through the contours and draw them on the image
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 5)

#     # Show the image with contours
#     cv2.imshow("Text Contours", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# def find_text_positions(image_path):
#     # Read the image
#     image = cv2.imread(image_path)

#     # Convert the image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Get OCR output with bounding box information
#     data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

#     # Iterate through the OCR output and extract the text positions
#     for i, text in enumerate(data["text"]):
#         if text.strip():
#             x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
#             rect_img = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
#             print(f"Text: {text}, Position: ({x}, {y}, {w}, {h})")
#     cv2.imshow(f"{text}", rect_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# def detect_vertical_lines_P(image_path):
#     # Read the image
#     image = cv2.imread(image_path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     cv2.imshow("Vertical Lines Gray", gray)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     # Apply a slight Gaussian blur to the image
#     blurred = cv2.GaussianBlur(gray, (7, 7), 0)
#     cv2.imshow("Vertical Lines Blurred", blurred)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     # Perform Canny edge detection
#     edges = cv2.Canny(blurred, 10, 300)
#     cv2.imshow("Vertical Lines Edges", edges)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     # Define a threshold angle for vertical lines (in degrees)
#     angle_threshold = 5  # Adjust this value based on your requirements

#     # Perform HoughLinesP transformation to detect lines
#     lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 2, minLineLength=1, maxLineGap=10)
#     # cv2.imshow("Vertical Lines Lines", lines)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#     print(len(lines))
#     # Iterate through the lines and draw only vertical lines
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
#             if 90 - angle_threshold <= angle <= 90 + angle_threshold:
#                 cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

#     # Show the image with detected vertical lines
#     cv2.imshow("Vertical Lines", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# def detect_vertical_lines(image_path):
#     # Read the image
#     image = cv2.imread(image_path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Apply a slight Gaussian blur to the image
#     blurred = cv2.GaussianBlur(gray, (5, 5), 1)

#     # Perform Canny edge detection
#     edges = cv2.Canny(blurred, 50, 300)

#     # Define a threshold angle for vertical lines (in degrees)
#     angle_threshold = 1  # Adjust this value based on your requirements

#     # Perform HoughLines transformation to detect lines
#     lines = cv2.HoughLines(edges, 1, np.pi / 90, 2)

#     # Iterate through the lines and draw only vertical lines
#     if lines is not None:
#         for line in lines:
#             rho, theta = line[0]
#             angle = theta * 90 / np.pi
#             if 90 - angle_threshold <= angle <= 90 + angle_threshold:
#                 a = np.cos(theta)
#                 b = np.sin(theta)
#                 x0 = a * rho
#                 y0 = b * rho
#                 x1 = int(x0 + 1000 * (-b))
#                 y1 = int(y0 + 1000 * (a))
#                 x2 = int(x0 - 1000 * (-b))
#                 y2 = int(y0 - 1000 * (a))
#                 cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

#     # Show the image with detected vertical lines
#     cv2.imshow("Vertical Lines", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# def vline_detector(image_path):
#     image = cv2.imread(image_path)
#     image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
#     cv2.imwrite('data/image/rotated_img.png',image)
#     image = detect_vertical_lines('data/image/rotated_img.png')
#     # cv2.imshow("Vertical Lines", image)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#     return image
    
# Call the function with the path to your image
# detect_vertical_lines(img_path)

# Call the function with the path to your image
# detect_vertical_lines_P(img_path)

# Call the function with the path to your image
# img = detect_horizontal_lines(img_path)
# cv2.imwrite('data/image/horzline.png',img=img)
# cv2.imshow("Horizontal Lines", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# image = detect_vertical_lines(rot_img_path)
# cv2.imwrite('data/image/vertline.png',img=img)


# Call the function with the path to your image
# find_text_positions(img_path)
# Call the function with the path to your image
# draw_contours_around_text(img_path)
# # df = pd.DataFrame(columns=['x', 'y', 'r', 'g', 'b'])
# img = cv2.imread(img_path)
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# print(img[200:205])
# print(img.shape)

# height, width, channels = img_rgb.shape
# list_pixel_color =[]
# for y in range(height):
#     print(f'row: {y}')
#     for x in range(width):
#         pixel = img_rgb[y, x]
#         if pixel[0] < 255 or pixel[1] <255 or pixel[2] <255 :
#             # print(pixel)
#             r, g, b = img_rgb[y, x]
#             list_pixel_color.append({'x': x, 'y': y, 'r': r, 'g': g, 'b': b})
            
#         # You can process the pixel values here (e.g., change color or apply a condition)

# # Plot the image using matplotlib
# df = pd.DataFrame(list_pixel_color)
# df.to_csv('data/csv/pixels.csv')
# plt.imshow(img_rgb)
# plt.xticks([]), plt.yticks([])  # Remove x and y axis ticks
# plt.show()


def detect_horizontal_lines(image_path):
    # Read the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a slight Gaussian blur to the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Define a threshold angle for horizontal lines (in degrees)
    angle_threshold = 10  # Adjust this value based on your requirements

    # Perform HoughLinesP transformation to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=3)

    # Iterate through the lines and draw only horizontal lines
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if -angle_threshold <= angle <= angle_threshold:
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show the image with detected horizontal lines

    return image

def detect_vertical_lines(image_path):
    # Read the image
    image = cv2.imread(image_path)
    cv2.imshow("Vertical Lines Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Vertical Lines Gray", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # Apply a slight Gaussian blur to the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imshow("Vertical Lines Blurred", blurred)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # Perform Canny edge detection
    edges = cv2.Canny(blurred, 50, 350)
    edges = 255 - edges
    cv2.imshow("Vertical Lines Edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    vert_kernel = np.ones((5,1),np.uint8)


    verticalLines = cv2.erode(edges, kernel=vert_kernel, iterations=2)
    cv2.imwrite("data/image/vertical_img_temp1.jpg",verticalLines)
    verticalLines = cv2.dilate(verticalLines, vert_kernel, iterations=2)
    cv2.imshow('Vertical Lines', verticalLines)
    cv2.waitKey(0)
    cv2.imwrite("data/image/vertical_img_temp2.jpg",verticalLines)

    # Define a threshold angle for horizontal lines (in degrees)
    angle_threshold = 20  # Adjust this value based on your requirements

    # Perform HoughLinesP transformation to detect lines
    lines = cv2.HoughLinesP(edges,1, np.pi / 180, 100, minLineLength=300, maxLineGap=5)

    # Iterate through the lines and draw only horizontal lines
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if -angle_threshold <= angle <= angle_threshold:
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show the image with detected horizontal lines

    return image


img = detect_horizontal_lines(img_path)
cv2.imwrite('data/image/horzline.png',img=img)
image = detect_vertical_lines(rot_img_path)
cv2.imwrite('data/image/vertline.png',img=image)