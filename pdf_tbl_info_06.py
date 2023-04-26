import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, color, morphology
import cv2, pytesseract


img_path = 'data/image/enhanced.png'




# Read image
image = cv2.imread(img_path)

# Convert image to grayscale
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# Use canny edge detection
edges = cv2.Canny(gray,50,150,apertureSize=3)


# Apply HoughLinesP method to
# to directly obtain line end points
lines_list =[]
lines = cv2.HoughLinesP(
			edges, # Input edge image
			3, # Distance resolution in pixels
			np.pi/180, # Angle resolution in radians
			threshold=200, # Min number of votes for valid line
			minLineLength=15, # Min allowed length of line
			maxLineGap=1 # Max allowed gap between line for joining them
			)

# Iterate over points
for points in lines:
	# Extracted points nested in the list
	x1,y1,x2,y2=points[0]
	# Draw the lines joing the points
	# On the original image
	cv2.line(image,(x1,y1),(x2,y2),(0,255,0),3)
	# Maintain a simples lookup list for points
	lines_list.append([(x1,y1),(x2,y2)])
	
# Save the result image
cv2.imwrite('data/image/detectedLines.png',image)
# cv2.imshow("Lines", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

