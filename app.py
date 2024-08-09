from flask import Flask, request, render_template, redirect, url_for
import os
import pytesseract
from PIL import Image
import numpy as np
import cv2
import mysql.connector
from mysql.connector import Error

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def is_ktp(image_path):
    image = Image.open(image_path)
    image_np = np.array(image)

    # Check if the image is grayscale (single channel)
    if len(image_np.shape) == 2 or image_np.shape[2] == 1:
        # Convert grayscale image to RGB by duplicating the single channel
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)

    # Convert the RGB image to HSV
    image_hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([140, 255, 255])

    mask = cv2.inRange(image_hsv, lower_blue, upper_blue)
    blue_ratio = cv2.countNonZero(mask) / (image_hsv.shape[0] * image_hsv.shape[1])

    color_check = blue_ratio > 0.1

    text = pytesseract.image_to_string(image)
    ktp_keywords = ["REPUBLIK INDONESIA", "NIK", "Nama", "Tempat/Tgl Lahir", "Alamat", "RT/RW", "Kel/Desa", "Kecamatan", "Agama", "Status Perkawinan", "Pekerjaan", "Kewarganegaraan"]

    text_check = any(keyword.lower() in text.lower() for keyword in ktp_keywords)

    if text_check and color_check:
        return "VALID", text
    elif not text_check:
        return "TIDAK VALID", "Text tidak sesuai dengan format KTP"
    elif not color_check:
        return "TIDAK VALID", "Warna background tidak sesuai dengan KTP"
    else:
        return "TIDAK VALID", "Gagal mengenali KTP"

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

    except Error as e:
        print(f"Error while connecting to MySQL: {e}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            is_valid, details = is_ktp(filepath)
            save_to_database(file.filename, is_valid)
            return render_template('result.html', filename=file.filename, is_valid=is_valid, details=details)
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run(debug=True)
