
from paddleocr import PaddleOCR, draw_ocr # main OCR dependencies
from matplotlib import pyplot as plt # plot images
import cv2 #opencv
import os # folder directory navigation
import numpy as np

# Setup model
ocr_model = PaddleOCR(lang='en')
# Image path
img_path = 'data/image/enhanced.png'
# Run the ocr method on the ocr model
result = ocr_model.ocr(img_path)
# print(result)
boxes = [res[0] for res in result]
image = cv2.imread(img_path)
# # Extracting detected components
# boxes = [res[0] for res in result] # 
# texts = [res[1][0] for res in result]
# scores = [res[1][1] for res in result]
# # Specifying font path for draw_ocr method
# font_path = os.path.join('PaddleOCR', 'doc', 'fonts', 'latin.ttf')

# # Import our image - drug 1/2/3
# # imports image
# img = cv2.imread(img_path) 

# # reorders the color channels
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
# # Visualize our image and detections
# # resizing display area
# plt.figure(figsize=(15,15))

# # draw annotations on image
# annotated = draw_ocr(img, boxes, texts, scores, font_path=font_path) 

# # show the image using matplotlib
# plt.imshow(annotated) 
def draw_box(image_path, points):
    # Read the image
    

    # Convert the points to a numpy array
    points = np.array(points, dtype=np.int32)

    # Draw the box using cv2.polylines()
    cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)
    return image 

for box in boxes:
    # Display the image with the box using OpenCV
    image = draw_box(image,box)
    
cv2.imshow("Image with Box", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
    
    