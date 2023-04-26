# import pytesseract
# from pytesseract import Output
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import numpy as np
import sys
import cv2 as cv
import os

img = cv2.imread('data/image/enhanced.png')
vert_kernel = np.ones((20,1),np.uint8)


verticalLines = cv2.erode(img, kernel=vert_kernel, iterations=1)
cv2.imshow('Vertical Lines', verticalLines)
cv2.waitKey(0)