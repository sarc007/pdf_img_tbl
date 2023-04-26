import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, color, morphology
import cv2, pytesseract


img_path = 'data/image/sample01.png'

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
            rect_img = cv2.rectangle(image, (x-2, y-2), (x + w+2, y + h+2), (255, 255, 255), cv2.FILLED)
            print(f"Text: {text}, Position: ({x}, {y}, {w}, {h})")
    return rect_img
    
def detect_vertical_lines_scikit(image):
    # Read the image
    # image = io.imread(image_path)
    gray = color.rgb2gray(image)

    # Apply the Sobel operator in the x-direction
    sobel_x = filters.sobel_v(gray)

    # Threshold the Sobel response to enhance vertical lines
    threshold = 0.001  # Adjust this value based on your requirements
    vertical_lines_mask = (sobel_x > threshold)

    # Perform morphological operations to clean up the image
    selem = morphology.rectangle(1, 5)  # Adjust the size based on your requirements
    vertical_lines_mask = morphology.dilation(vertical_lines_mask, selem)
    vertical_lines_mask = morphology.erosion(vertical_lines_mask, selem)

    # Convert the binary mask to an 8-bit grayscale image
    vertical_lines_image = (vertical_lines_mask * 255).astype(np.uint8)

    return vertical_lines_image




no_text_image =  find_text_positions(img_path)

# Call the function with the path to your image
vertical_lines_image = detect_vertical_lines_scikit(no_text_image)
# Call the function with the path to your image



# Display the detected vertical lines using OpenCV
cv2.imshow("Detected Vertical Lines", vertical_lines_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
