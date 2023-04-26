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

def vertical_crop_image(h_images,stem,df_with_h_coordinates):
	veri_crop_image = []
	veri_coor_list = []
	hori_coor_list = []

	for src, idx in zip(h_images,df_with_h_coordinates.itertuples()):
		# print(type(src))
		# print(idx)
		a = idx.x1
		b = idx.y1
		c = idx.x2
		d = idx.y2
		hori_coor_list.append([a, b, c, d])

		if len(src.shape) != 2:
			gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
		else:
			gray = src
		cv2.imwrite("data/image/horz/h_image.jpg",gray)
		gray = cv.bitwise_not(gray)
		bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)
		vertical = np.copy(bw)
		# Identify vertical lines in image-----------------------
		rows = vertical.shape[0]
		columns = vertical.shape[1]
		if (rows < 700) and (columns < 2000):
			verticalsize = rows
		elif (rows < 300) and (columns > 2000):
			verticalsize = rows
		else:
			verticalsize = rows // 10

		verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalsize))
		vertical = cv.erode(vertical, verticalStructure)
		vertical = cv.dilate(vertical, verticalStructure)

		# crating and processing Pandas Dataframe for identified vertical lines --------------------
		vcontours, hierarchy = cv2.findContours(vertical, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  # len of contous is total lines detected
		x_veri = []
		y_veri = []
		w_veri = []
		h_veri = []
		for v_cnt in vcontours:
			x, y, w, h = cv2.boundingRect(v_cnt)
			x_veri.append(x)
			y_veri.append(y)
			w_veri.append(w)
			h_veri.append(h)

		df_veri = pd.DataFrame([x_veri, y_veri, w_veri, h_veri]).transpose()
		df_veri.columns = ["x_veri", "y_veri", "w_veri", "h_veri"]
		# NEW code----------------------------------
		veri_coor_list.append([a, b, c, d])
		print("horizontal image----------------")
		print(veri_coor_list)
		# if Dataframe has more than 2 lines crop image------------------------
		if len(df_veri) > 1:
			df_veri["x2"] = df_veri["x_veri"] + df_veri["w_veri"]
			df_veri["y2"] = df_veri["y_veri"] + df_veri["h_veri"]
			df_veri = df_veri.sort_values(by=["y_veri", "y2"])
			df_veri = df_veri.sort_values(by=["x_veri"])
			df_veri.columns = ['x1', 'y1', 'w_veri', 'h_veri', 'x2', 'y2']
			df_veri["x1_n"] = df_veri["x1"].rolling(2).min()
			df_veri["x2_n"] = df_veri["x2"].rolling(2).max()
			df_veri = df_veri.dropna()
			counter = 1
			crop_regions = []
			for idx in df_veri.itertuples():
				crop = src[15:-15, int(idx.x1_n+15): int(idx.x2_n-15)]
				# cv2.imwrite(stem + '_out_{}.jpg'.format(counter))
				veri_crop_image.append(crop)
				counter = counter + 1
				veri_coor_list.append([idx.x1_n,idx.y1,idx.x2_n,idx.y2])
				print("vertical crop iteration--------------------")
				print(veri_coor_list)
		else:
			veri_crop_image.append(src)
			df_veri_new = df_veri
			df_veri_new["x2"] = df_veri_new["x_veri"] + df_veri_new["w_veri"]
			df_veri_new["y2"] = df_veri_new["y_veri"] + df_veri_new["h_veri"]
			veri_coor_list.append([df_veri_new["x_veri"],df_veri_new["x2"],df_veri_new["y_veri"],df_veri_new["y2"]])
			print("no need to crop--------")
			print(veri_coor_list)
		# df_veri.to_csv("C://Users//priya.lotankar//sept_2019//DPI_images//New_folder//df_veri.csv")
	return veri_crop_image , df_veri, veri_coor_list

def horizontal_crop_image(out_image,stem,df):
	hori_crop_image = []

	if len(out_image.shape) != 2:
		gray = cv.cvtColor(out_image, cv.COLOR_BGR2GRAY)
	else:
		gray = out_image

	gray = cv.bitwise_not(gray)
	bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)

	# CREATING dATAFRAME FOR HORIZONTAL LINES--------------------------------------------------
	horizontal = np.copy(bw)
	cols = horizontal.shape[1]
	horizontal_size = cols // 15

	horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
	horizontal = cv.erode(horizontal, horizontalStructure)
	horizontal = cv.dilate(horizontal, horizontalStructure)

	hcontours,hierarchy = cv2.findContours(horizontal,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	x_hori = []
	y_hori = []
	w_hori = []
	h_hori = []
	for h_cnt in hcontours:
		x,y,w,h = cv2.boundingRect(h_cnt)
		x_hori.append(x)
		y_hori.append(y)
		w_hori.append(w)
		h_hori.append(h)

	df_hori = pd.DataFrame([x_hori,y_hori,w_hori,h_hori]).transpose()
	if len(x_hori)>0 and len(y_hori)>0 and  len(w_hori)>0 and len(h_hori)>0:
		df_hori.columns = ["x_hori","y_hori","w_hori","h_hori"]
		df_hori["x2"] = df_hori["x_hori"] + df_hori["w_hori"]
		df_hori["y2"] = df_hori["y_hori"] + df_hori["h_hori"]
		df_hori = df_hori.sort_values(by =["x_hori","x2"])
		df_hori["diff_x"] = df_hori["x2"] - df_hori["x_hori"]

		#EXTRACT HORIZONTAL LINES------------------------------------------------------
		df_hori = df_hori.sort_values(by=["y_hori"])
		df_hori["w_hori - min"] = round((df_hori["w_hori"]-df_hori["w_hori"].min())/df_hori["w_hori"].mean()*100,0)
		df_hori["x1_"] = df_hori["x_hori"].rolling(2).min()
		df_hori["y1_"] = df_hori["y_hori"].rolling(2).min()
		df_hori["x2_"] = df_hori["x2"].rolling(2).min()
		df_hori["y2_"] = df_hori["y2"].rolling(2).max()
		df_hori = df_hori.reset_index()
		df_hori["x1_"][0] = df_hori["x1_"][1]
		df_hori["y1_"][0] = df_hori["y1_"][1]
		df_hori["x2_"][0] = df_hori["x2_"][1]
		df_hori["y2_"][0] = df_hori["y2_"][1]
		df_hori["x2_per_change"] = round((df_hori["x2"] - df_hori["x2_"])/(df_hori["x2"])*100,0)
		df_hori_revised = df_hori[df_hori["x2_per_change"]>10]
		df_hori = df_hori.drop(columns=["w_hori","h_hori","diff_x","w_hori - min","x2_per_change"])
		df_hori_revised = df_hori_revised.drop(columns=["w_hori","h_hori","diff_x","w_hori - min","x2_per_change"])
		df_hori_revised["x1_n"] = df_hori_revised["x2_"] + 1
		df_hori_revised["y1_n"] = df_hori_revised["y_hori"].rolling(2).min()
		df_hori_revised["x2_n"] = df_hori_revised["x2"]
		df_hori_revised["y2_n"] = df_hori_revised["y2"].rolling(2).max()
		df_hori_revised = df_hori_revised.drop(columns=["x1_","x2_","y1_","y2_"])
		df_hori_revised.columns = ['index', 'x_hori', 'y_hori', 'x2', 'y2', 'x1_', 'y1_', 'x2_','y2_']
		df_crop = pd.concat([df_hori,df_hori_revised])
		df_crop.columns = ['index', 'x_hori', 'y_hori', 'x2_old', 'y2_old', 'x1', 'y1', 'x2', 'y2']
		df_crop = df_crop.sort_values(by="y1")
		df_crop = df_crop.reset_index()
		df_crop = df_crop.dropna()
		df_crop = df_crop[1:]

		counter = 1
		crop_regions = []
		for idx in df_crop.itertuples():
			crop = out_image[int(idx.y1) : int(idx.y2) , int(idx.x1) : int(idx.x2)]
			# file_name = stem + "H_crop_{}.jpg".format(counter)
			hori_crop_image.append(crop)
			# cv2.imwrite(file_name,crop)
			counter = counter + 1
	else:
		return None, None
	return hori_crop_image, df_crop

def hv_lines(dir):
	#READING IMAGE FILES FROM FOLDER----------------------------------------
	image_blocks = {}
	original_image = {}
	h_crop_image = []
	for root, dirs, files in os.walk(dir):
		for file in files:
			path = os.path.join(root, file)
			[stem, ext] = os.path.splitext(path)
			if (ext == '.jpg') or (ext == ".png"):
				# print("Processing: " + file)
				# Horizontal and vertical lines detection----
				src = cv.imread(path, cv.IMREAD_COLOR)
				original_image[file] = src
				# Transform source image to gray if it is not already
				if len(src.shape) != 2:
					gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
				else:
					gray = src

				gray = cv.bitwise_not(gray)
				bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)

				horizontal = np.copy(bw)
				vertical = np.copy(bw)
				cols = horizontal.shape[1]
				horizontal_size = cols // 15
				if horizontal_size == 0: horizontal_size=1
				horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
				horizontal = cv.erode(horizontal, horizontalStructure)
				horizontal = cv.dilate(horizontal, horizontalStructure)

				rows = vertical.shape[0]
				verticalsize = rows // 30
				if verticalsize ==0: verticalsize =1
				verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalsize))
				vertical = cv.erode(vertical, verticalStructure)
				vertical = cv.dilate(vertical, verticalStructure)

				hcontours, hierarchy = cv2.findContours(horizontal, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
				vcontours, hierarchy = cv2.findContours(vertical, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
				# Creating and processing dataframe-----------------------------------
				x_hori = []
				y_hori = []
				w_hori = []
				h_hori = []
				for h_cnt in hcontours:
					x, y, w, h = cv2.boundingRect(h_cnt)
					x_hori.append(x)
					y_hori.append(y)
					w_hori.append(w)
					h_hori.append(h)
					out = cv2.line(src, (x, y), (x + w, y + h), (0, 0, 0), 4)  # changes to 4

				x_veri = []
				y_veri = []
				w_veri = []
				h_veri = []
				for v_cnt in vcontours:
					x, y, w, h = cv2.boundingRect(v_cnt)
					x_veri.append(x)
					y_veri.append(y)
					w_veri.append(w)
					h_veri.append(h)
					out = cv2.line(src, (x, y), (x + w, y + h), (0, 0, 0), 4)

				df_hori = pd.DataFrame([x_hori, y_hori, w_hori, h_hori]).transpose()
				df_hori.columns = ["x_hori", "y_hori", "w_hori", "h_hori"]

				df_veri = pd.DataFrame([x_veri, y_veri, w_veri, h_veri]).transpose()
				df_veri.columns = ["x_veri", "y_veri", "w_veri", "h_veri"]

				df_hori["x2"] = df_hori["x_hori"] + df_hori["w_hori"]
				df_veri["x2"] = df_veri["x_veri"] + df_veri["w_veri"]

				df_hori["y2"] = df_hori["y_hori"] + df_hori["h_hori"]
				df_veri["y2"] = df_veri["y_veri"] + df_veri["h_veri"]

				df_hori = df_hori.sort_values(by=["x_hori", "x2"])
				df_veri = df_veri.sort_values(by=["y_veri", "y2"])

				df_hori["diff_x"] = df_hori["x2"] - df_hori["x_hori"]
				df_veri["diff_x"] = df_veri["x2"] - df_veri["x_veri"]

				# Add missing border------------------------------------------------------------------------------
				x1 = min(df_hori["x_hori"].tolist())
				x2 = max(df_hori["x2"].tolist())
				y1 = min(df_hori["y_hori"].tolist())
				y2 = max(df_veri["y2"].tolist())

				out_image = cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 0), 4)
				cv2.imwrite(stem + "OUT_IMAGE.jpg",out_image)
				h_crop_image.append(out_image) # save each images as list of array
				crop_image_h, df_with_h_coordinates = horizontal_crop_image(out_image,stem,df_hori)
				crop_image_v, df_with_v_coordinates, vertical_coor_list = vertical_crop_image(crop_image_h,stem,df_with_h_coordinates)
				image_blocks[file] = crop_image_v
	counter = 1
	for i in image_blocks[file]:
		if i.shape()[0] > 0:
			cv2.imwrite(stem + '_out_{}.jpg'.format(counter), i)
			counter = counter + 1
	vertical_df = pd.DataFrame(vertical_coor_list,columns=["x1","x2","y1","y2"])
	vertical_df.to_csv("data/csv/vertical_df.csv")
	df_with_h_coordinates.to_csv("data/csv/horizontal_df.csv")
	return image_blocks,original_image, df_with_h_coordinates, df_with_v_coordinates

image_block,original_img, df_h_coordiantes, df_v_coordinates = hv_lines("data/image/horz")
#
# f  = open("abc.txt","w")
# f.write(str(image_block))
# # print(image_block,file = f)
# f.close()