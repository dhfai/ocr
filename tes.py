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
PROCESSED_FOLDER = 'processed_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Ensure the upload and processed image folders exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

# List of allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply thresholding to create a binary image
    _, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Save the processed image
    processed_image_path = os.path.join(PROCESSED_FOLDER, os.path.basename(image_path))
    cv2.imwrite(processed_image_path, thresh)
    
    return thresh, processed_image_path

def extract_text(image_path):
    preprocessed_image, processed_image_path = preprocess_image(image_path)
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(preprocessed_image, config=custom_config)
    return text, processed_image_path

def is_ktp(image_path):
    image = Image.open(image_path)
    image_np = np.array(image)

    # Check if the image is grayscale (single channel)
    if len(image_np.shape) == 2 or image_np.shape[2] == 1:
        # Convert grayscale image to RGB by duplicating the single channel
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)

    # Convert the RGB image to HSV
    image_hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

    # Define color ranges for blue and white backgrounds
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([140, 255, 255])

    lower_white = np.array([0, 0, 168])  # Lower threshold for value
    upper_white = np.array([172, 111, 255])  # Adjust saturation and value range

    # Create masks for blue and white backgrounds
    blue_mask = cv2.inRange(image_hsv, lower_blue, upper_blue)
    white_mask = cv2.inRange(image_hsv, lower_white, upper_white)

    # Calculate the ratio of the specific color in the image
    blue_ratio = cv2.countNonZero(blue_mask) / (image_hsv.shape[0] * image_hsv.shape[1])
    white_ratio = cv2.countNonZero(white_mask) / (image_hsv.shape[0] * image_hsv.shape[1])

    # Check if either blue or white ratio is significant
    color_check = blue_ratio > 0.1 or white_ratio > 0.1

    if not color_check:
        return "TIDAK VALID", "Warna background tidak sesuai dengan KTP", None


    text, processed_image_path = extract_text(image_path)
    print(text)
    ktp_keywords = ["REPUBLIK INDONESIA", "NIK", "Nama", "Tempat/Tgl Lahir", "Alamat", "RT/RW", "Kel/Desa", "Kecamatan", "Agama", "Status Perkawinan", "Pekerjaan", "Kewarganegaraan"]


    text_check = any(keyword.lower() in text.lower() for keyword in ktp_keywords)


    if text_check:
        return "VALID", text, processed_image_path
    else:
        return "TIDAK VALID", "Text tidak sesuai dengan format KTP", processed_image_path

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
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            is_valid, details, processed_image_path = is_ktp(filepath)
            save_to_database(file.filename, is_valid)
            return render_template('result.html', filename=file.filename, is_valid=is_valid, details=details, processed_image_path=processed_image_path)
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run(debug=True)
