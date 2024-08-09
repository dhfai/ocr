import os
import pytesseract
from PIL import Image
import numpy as np
import cv2
import mysql.connector
from mysql.connector import Error

def is_ktp(image_path):
    # Load the image from the uploaded file
    image = Image.open(image_path)
    
    # Convert image to numpy array and then to HSV
    image_np = np.array(image)
    image_hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

    # Define range for blue color in HSV
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([140, 255, 255])

    # Create a mask for blue color
    mask = cv2.inRange(image_hsv, lower_blue, upper_blue)
    blue_ratio = cv2.countNonZero(mask) / (image_hsv.shape[0] * image_hsv.shape[1])

    # Consider the image to have a blue color if more than 10% of the pixels are blue
    color_check = blue_ratio > 0.1

    # Perform OCR on the image
    text = pytesseract.image_to_string(image)

    # Define common keywords found in KTP
    ktp_keywords = ["REPUBLIK INDONESIA", "NIK", "Nama", "Tempat/Tgl Lahir", "Alamat", "RT/RW", "Kel/Desa", "Kecamatan", "Agama", "Status Perkawinan", "Pekerjaan", "Kewarganegaraan"]

    # Check if any of the keywords are found in the extracted text
    text_check = any(keyword.lower() in text.lower() for keyword in ktp_keywords)

    # The image is considered KTP if both text and color checks are passed
    return "VALID" if text_check and color_check else "TIDAK VALID"

def save_to_database(filename, is_valid):
    try:
        connection = mysql.connector.connect(
            host='if.unismuh.ac.id',
            database='ocr',
            user='root',
            password='mariabelajar',
            port=3388
        )

        if connection.is_connected():
            cursor = connection.cursor()
            insert_query = """INSERT INTO ktp_validation (filename, is_valid) VALUES (%s, %s)"""
            record = (filename, is_valid)
            cursor.execute(insert_query, record)
            connection.commit()
            print(f"{filename} saved to database successfully")

    except Error as e:
        print(f"Error while connecting to MySQL: {e}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")

def process_dataset(folder_path):
    results = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            image_path = os.path.join(folder_path, filename)
            result = is_ktp(image_path)
            results[filename] = result
            save_to_database(filename, result)
    return results

# Path to the folder containing images
folder_path = "data"  # Ganti path ini dengan path yang sesuai

# Process the dataset and print the results
results = process_dataset(folder_path)
for filename, is_ktp_image in results.items():
    print(f"{filename}: {is_ktp_image}")
