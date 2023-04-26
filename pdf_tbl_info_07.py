import cv2
import numpy as np


img_path = 'data/image/sample01.png'

def enhance_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the image (optional, depending on your needs)
    scale_percent = 300  # percent of original size
    width = int(gray.shape[1] * scale_percent / 100)
    height = int(gray.shape[0] * scale_percent / 100)
    resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)

    # Denoise the image using Non-local Means Denoising
    denoised = cv2.fastNlMeansDenoising(resized, h=10, templateWindowSize=7, searchWindowSize=21)

    # Apply histogram equalization to improve contrast
    equalized = cv2.equalizeHist(denoised)

    # Enhance contrast and brightness
    alpha = 1.2  # contrast control (1.0-3.0)
    beta = 30  # brightness control (0-100)
    adjusted = cv2.convertScaleAbs(equalized, alpha=alpha, beta=beta)

    # Apply a sharpening filter
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(adjusted, -1, kernel)

    return sharpened

# Call the function with the path to your image
enhanced_image = enhance_image(img_path)

# Display the enhanced image using OpenCV
# cv2.imshow("Enhanced Image", enhanced_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite('data/image/enhanced.png',enhanced_image)