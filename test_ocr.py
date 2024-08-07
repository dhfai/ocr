import os
import cv2
import pytesseract
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import numpy as np

def display(img, cmap='gray'):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')

# Load image
img_path = "ktp.png"
img = cv2.imread(img_path)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Increase contrast
contrast = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)

# Apply adaptive thresholding
threshed = cv2.adaptiveThreshold(contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Apply morphological operations to remove noise
kernel = np.ones((2, 2), np.uint8)
morph = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)
morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

# Show the processed image
plt.imshow(morph, cmap='gray')
plt.show()

# Extract text data
text1 = pytesseract.image_to_data(morph, output_type='data.frame')
text2 = pytesseract.image_to_string(morph, lang="ind")
print("Extracted Text:\n", text2)

# Filter out rows with confidence level of -1
text = text1[text1.conf != -1]

# Group text by block number
lines = text.groupby('block_num')['text'].apply(list)
conf = text.groupby(['block_num'])['conf'].mean()

# Set display options for pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', text.shape[0] + 1)

# Display extracted text data
print("text1 : \n")
print(text1, "\n\n")

print("text : \n")
print(text, "\n\n")

print("lines : \n")
for i in range(len(lines)):
    print("level", i, ": ", lines.iloc[i])

print("\n\n conf : \n")
print(conf)

# Function to validate the structure of KTP
def validate_ktp(extracted_text):
    required_fields = ["NIK", "Nama", "Tempat/Tgl Lahir", "Alamat", "RT/RW", "Kel/Desa", "Kecamatan", "Agama", "Status Perkawinan", "Pekerjaan", "Kewarganegaraan", "Berlaku Hingga"]
    
    valid = True
    for field in required_fields:
        if field not in extracted_text:
            valid = False
            print(f"Missing field: {field}")
            break
    
    return valid

# Validate the extracted text
is_valid = validate_ktp(text2)
if is_valid:
    print("KTP is valid.")
else:
    print("KTP is invalid.")

# Draw bounding boxes on the image
from pytesseract import Output

img = cv2.imread(img_path)

d = pytesseract.image_to_data(img, output_type=Output.DICT)
print(d.keys())

n_boxes = len(text1['text'])
for i in range(n_boxes):
    if int(text1['conf'][i]) > 60:
        (x, y, w, h) = (text1['left'][i], text1['top'][i], text1['width'][i], text1['height'][i])
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

display(img)
