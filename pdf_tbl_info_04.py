import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, color
import cv2,pytesseract



img_path = 'data/image/sample01.png'


def detect_vertical_lines(image_path):
    # Read the image
    image = io.imread(image_path)
    gray = color.rgb2gray(image)
    gray = 255-gray

    # Apply the Sobel operator in the x-direction
    sobel_x = filters.sobel_v(gray)

    # Threshold the Sobel response to enhance vertical lines
    threshold = 0.0001  # Adjust this value based on your requirements
    vertical_lines_mask = (sobel_x > threshold)

    # Create an empty black image with the same shape as the input image
    vertical_lines_image = np.zeros_like(image)

    # Draw the detected vertical lines onto the black image
    vertical_lines_image[vertical_lines_mask] = image[vertical_lines_mask]

    return vertical_lines_image

# Call the function with the path to your image
vertical_lines_image = detect_vertical_lines(img_path)

# Convert the vertical_lines_image array to an OpenCV-compatible image
vertical_lines_cv2 = vertical_lines_image.astype(np.uint8)

def find_text_positions(image_path):
    # Read the image
    image = cv2.imread(image_path)

    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a slight Gaussian blur to the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)


    # Get OCR output with bounding box information
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    # Iterate through the OCR output and extract the text positions
    rect_img_array = []
    for i, text in enumerate(data["text"]):
        if text.strip() and text.strip() not in ['i','I','l','L','T','t','|','='] :
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            # rect_img_array.append(cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1))
            # cv2.imshow(f"{text}", rect_img_array[-1])
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()                
            rect_img = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), cv2.FILLED)
            print(f"Text: {text}, Position: ({x}, {y}, {w}, {h})")
    cv2.imshow(f"{text}", rect_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_vertical_lines_P(image):
    # Read the image
    # image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Vertical Lines Gray", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Apply a slight Gaussian blur to the image
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    cv2.imshow("Vertical Lines Blurred", blurred)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Perform Canny edge detection
    edges = cv2.Canny(blurred, 10, 300)
    cv2.imshow("Vertical Lines Edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Define a threshold angle for vertical lines (in degrees)
    angle_threshold = 5  # Adjust this value based on your requirements

    # Perform HoughLinesP transformation to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 2, minLineLength=1, maxLineGap=10)
    # cv2.imshow("Vertical Lines Lines", lines)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print(len(lines))
    # Iterate through the lines and draw only vertical lines
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if 90 - angle_threshold <= angle <= 90 + angle_threshold:
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show the image with detected vertical lines
    cv2.imshow("Vertical Lines", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Call the function with the path to your image
# detect_vertical_lines_P(vertical_lines_cv2)

find_text_positions(img_path)


# Display the detected vertical lines using OpenCV
cv2.imshow("Detected Vertical Lines", vertical_lines_cv2)
cv2.waitKey(0)
cv2.destroyAllWindows()
