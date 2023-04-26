import cv2
import numpy as np
from pdf2image import convert_from_path
import os
pdf_path = 'data/pdf/pdf_from_shahbaz.pdf'
images = convert_from_path(pdf_path=pdf_path)

def enhance_image(image):
    # Read the image
    # image = cv2.imread(image_path)
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
def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


#Functon for extracting the box
def box_extraction(img_for_box_extraction_path, cropped_dir_path, page_dir_name):

    print("Reading image..")
    img = cv2.imread(img_for_box_extraction_path)  # Read the image
    img = enhance_image(img)
    # print(img[:100])
    (thresh, img_bin) = cv2.threshold(img, 128, 255,
                                      cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Thresholding the image

    # print("Storing binary image to Images/Image_bin1.jpg..")
    # cv2.imwrite("tmp/"+page_dir_name+"/Images/Image_bin1.jpg",img_bin)

    # Apply a slight Gaussian blur to the image
    blurred = cv2.GaussianBlur(img_bin, (5, 5), 0)

    # Perform Canny edge detection
    edges = cv2.Canny(blurred, 50, 350)

    img_bin = 255-edges  # Invert the image

    print("Storing binary image to Images/Image_bin.jpg..")
    img_bin =255 - img_bin
    cv2.imwrite("tmp/"+page_dir_name+"/Images/Image_bin.jpg",img_bin)

    print("Applying Morphological Operations..")
    # Defining a kernel length
    kernel_length = 70
    # print(kernel_length) 
    # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    # A kernel of (3 X 3) ones.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Morphological operation to detect verticle lines from an image
    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=1)
    cv2.imwrite("tmp/"+page_dir_name+"/Images/img_temp1.jpg",img_temp1)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=1)
    cv2.imwrite("tmp/"+page_dir_name+"/Images/verticle_lines.jpg",verticle_lines_img)

    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
    cv2.imwrite("tmp/"+page_dir_name+"/Images/horizontal_lines.jpg",horizontal_lines_img)

    # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
    alpha = 0.5
    beta = 1.0 - alpha
    # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
    img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # For Debugging
    # Enable this line to see verticle and horizontal lines in the image which is used to find boxes
    print("Binary image which only contains boxes: Images/img_final_bin.jpg")
    cv2.imwrite("tmp/"+page_dir_name+"/Images/img_final_bin.jpg",img_final_bin)
    # Find contours for image, which will detect all the boxes
    contours, hierarchy = cv2.findContours(
        img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Sort all the contours by top to bottom.
    (contours, boundingBoxes) = sort_contours(contours, method="left-to-right")

    print("Output stored in Output directiory!")

    idx = 0
    for c in contours:
        # Returns the location and width,height for every contour
        x, y, w, h = cv2.boundingRect(c)

        # If the box height is greater then 20, widht is >80, then only save it as a box in "cropped/" folder.
        if (w > 80 and h > 20) and w > 3*h:
            idx += 1
            new_img = img[y:y+h, x:x+w]
            cv2.imwrite(cropped_dir_path+str(idx) + '.png', new_img)

    # For Debugging
    # Enable this line to see all contours.
    # cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
    # cv2.imwrite("./Temp/img_contour.jpg", img)



#Input image path and out folder
mdir = os.path.join(os.curdir, 'data/img/')
if not os.path.exists(mdir):
    os.mkdir(mdir)
mdir = os.path.join(os.curdir, 'data/img/contours/')
if not os.path.exists(mdir):
    os.mkdir(mdir)
mdir = os.path.join(os.curdir, 'tmp/')
if not os.path.exists(mdir):
    os.mkdir(mdir)

page_number = 1
for image in images:
    image.save('tmp/tmp.png', 'png')
    if page_number <10:
        page_str = '00'+str(page_number)
    elif page_number<100 and page_number>9:
        page_str = '0'+str(page_number)
    else:
        page_str = str(page_number)
    page_str = 'page-' + page_str 
    directory_path = os.path.join(os.curdir, 'data/img/contours/' + page_str + '/')
    if not os.path.exists(directory_path):
        os.mkdir(directory_path)
    page_dir = os.path.join(os.curdir, 'tmp/'+page_str+'/')
    if not os.path.exists(page_dir):
        os.mkdir(page_dir)
    page_dir = os.path.join(os.curdir, 'tmp/'+page_str+'/Images/')
    if not os.path.exists(page_dir):
        os.mkdir(page_dir)

    
    page_number += 1
    box_extraction('tmp/tmp.png', cropped_dir_path= directory_path, page_dir_name= page_str)
    os.remove('tmp/tmp.png')