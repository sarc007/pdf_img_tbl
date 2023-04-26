
# import tabula

# # Read pdf into list of DataFrame
# pdf_path = 'data/pdf/pdf_from_shahbaz.pdf'
# dfs = tabula.read_pdf(pdf_path, pages='all')
# print(dfs)
# # from pdf2image import convert_from_path
# # from PIL import Image

# # # Set the PDF file path
# # pdf_path = 'data/pdf/pdf_from_shahbaz.pdf'

# # # Set the output image path
# # output_image_path = 'data/img/img_from_shahbaz.jpg'

# # # Convert the PDF to a list of PIL.Image objects
# # images = convert_from_path(pdf_path)

# # # Save the first page of the PDF as an image
# # # If you want to save all pages, you can loop through the images list
# # image = images[0]
# # image.save(output_image_path, 'JPEG')
# # import camelot

# # # Set the path to the PDF file
# # pdf_file = 'data/pdf/pdf_from_shahbaz.pdf'

# # # Extract tables from the PDF file
# # tables = camelot.read_pdf(pdf_file)

# # # Print the number of tables extracted
# # print(f"Total tables extracted: {tables.n}")

# # # Iterate through the tables and print their content as Pandas DataFrames
# # for i, table in enumerate(tables, start=1):
# #     print(f"\nTable {i}:")
# #     print(table.df)
# # import os
# # import camelot
# # import pytesseract
# # from pdf2image import convert_from_path
# # from PIL import Image
# # from io import BytesIO

# # # Set the paths to the input PDF and output CSV files
# # pdf_file = 'data/pdf/pdf_from_shahbaz.pdf'
# # output_file = 'data/csv/pdf_from_shahbaz.csv'

# # # Convert the PDF to a list of PIL.Image objects
# # images = convert_from_path(pdf_file)

# # # Perform OCR on the images using pytesseract
# # ocr_text = ''
# # for image in images:
# #     ocr_text += pytesseract.image_to_string(image)

# # # Save the OCR text to a temporary text file
# # with open('tmp/temp.txt', 'w', encoding='utf-8') as f:
# #     f.write(ocr_text)

# # # Extract tables from the text file using camelot-py
# # tables = camelot.read_text('tmp/temp.txt')

# # # Save the extracted tables as CSV
# # tables.export(output_file, f='csv')

# # # Remove the temporary text file
# # os.remove('tmp/temp.txt')

# # print("Table extraction completed.")

# # import os
# # import tabula
# # import pytesseract
# # from pdf2image import convert_from_path
# # from PIL import Image
# # from io import BytesIO

# # # Set the paths to the input PDF and output CSV files
# # pdf_file = 'data/pdf/pdf_from_shahbaz.pdf'
# # output_file = 'data/csv/pdf_from_shahbaz.csv'

# # # Convert the PDF to a list of PIL.Image objects
# # images = convert_from_path(pdf_file)

# # # Perform OCR on the images using pytesseract
# # ocr_text = ''
# # for image in images:
# #     ocr_text += pytesseract.image_to_string(image)

# # # Save the OCR text to a temporary text file
# # with open('tmp/temp.txt', 'w', encoding='utf-8') as f:
# #     f.write(ocr_text)

# # # Extract tables from the text file using tabula-py
# # tables = tabula.read_pdf('tmp/temp.txt', output_format='dataframe', pages='all', multiple_tables=True)

# # # Save the extracted tables as CSV
# # for i, table in enumerate(tables, start=1):
# #     table.to_csv(f'{output_file[:-4]}_table_{i}.csv', index=False)

# # # Remove the temporary text file
# # os.remove('tmp/temp.txt')

# # print("Table extraction completed.")
