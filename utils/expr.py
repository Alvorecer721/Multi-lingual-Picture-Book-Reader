from PIL import Image
import sys
from tesserocr import PyTessBaseAPI
import cv2
import pytesseract

testI = "D:/Individual_Project/individual_project/temp_folder_29qzv5gd/0.jpg"

#%%
column = Image.open(testI)
gray = column.convert('L')
blackwhite = gray.point(lambda x: 0 if x < 200 else 255, '1')
# blackwhite.save("code_bw.jpg")

# %%
with PyTessBaseAPI() as api:
    api.Init(lang='jpn')
    api.SetImageFile("D:/Individual_Project/individual_project/code_bw.jpg")
    print(api.GetUTF8Text())

# %%
img = cv2.imread("D:/Individual_Project/individual_project/code_bw.jpg")
img2 = cv2.imread(testI)
rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
cv2.imwrite("before.jpg", rgb)

# options = "-l {}".format('jpn')
# text = pytesseract.image_to_string(img, config=options)
# print(text)

# %%

# Grayscale, Gaussian blur, Otsu's threshold
image = cv2.imread(testI)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3, 3), 0)
thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)[1]

options = "-l {}".format('jpn')
text = pytesseract.image_to_string(blur, config=options)
print(text)

#%%
cv2.imwrite('wtf.jpg', thresh)

#%%
# Morph open to remove noise and invert image
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
# invert = 255 - opening

invert = 255 - thresh

# Perform text extraction
options = "-l {}".format('jpn')
data = pytesseract.image_to_string(invert, config=options)
print(data)

#%%
cv2.imwrite('wtf.jpg', opening)

#%%
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread(r"D:\Individual_Project\individual_project\utils\5.jpg", 0)
img = cv.medianBlur(img,5)
th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY,11,2)


cv.imwrite('AMT.jpg', th2)
