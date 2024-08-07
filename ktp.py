import cv2
import numpy as np
import pytesseract
import matploitlib.pyplot as plty
from PIL import image

img = = cv2.imread('ktp.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)

result = pytesseract.image_to_string((threshed), lang='ind')

for word in result.split('\n'):
    if "_" in word:
        word = word.replace("_", ":")