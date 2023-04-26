import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, color
import cv2
import pandas as pd

output_file = 'data/csv/vertical_lines.csv'
img_path = 'data/image/sample01.png'

def detect_vertical_lines(image_path):
    # Read the image
    image = io.imread(image_path)
    gray = color.rgb2gray(image)

    # Apply the Sobel operator in the x-direction
    sobel_x = filters.sobel_v(gray)

    # Threshold the Sobel response to enhance vertical lines
    threshold = 0.1  # Adjust this value based on your requirements
    vertical_lines = (sobel_x > threshold).astype(np.uint8)

    return vertical_lines

# Call the function with the path to your image
vertical_lines = detect_vertical_lines(img_path)
vertical_lines_cv2 = (vertical_lines * 255).astype(np.uint8)
# print(vertical_lines)
df = pd.DataFrame(list(vertical_lines))
df.to_csv(output_file)
cv2.imshow("Text Contours", vertical_lines_cv2)
cv2.waitKey(0)
cv2.destroyAllWindows()
# # Display the original image and the detected vertical lines
# fig, axes = plt.subplots(1, 2, figsize=(15, 5))
# axes[0].imshow(io.imread(img_path))
# axes[0].set_title("Original Image")
# axes[1].imshow(vertical_lines, cmap='Dark2')
# axes[1].set_title("Detected Vertical Lines")
# plt.show()
