import os
import cv2
import pytesseract
from pytesseract import Output
import numpy as np

# Path to tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# Function to check the background color of the image
def validate_background_color(image):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for the background color (light blue background)
    lower_blue = np.array([80, 20, 70], dtype=np.uint8)
    upper_blue = np.array([140, 255, 255], dtype=np.uint8)
    
    # Create mask to detect background color
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Calculate the percentage of the background color in the image
    blue_pixels = cv2.countNonZero(mask)
    total_pixels = image.shape[0] * image.shape[1]
    blue_pixel_percentage = (blue_pixels / total_pixels) * 100
    
    # If more than 50% of the pixels are blue, assume valid background
    if blue_pixel_percentage > 50:
        return True
    else:
        return False

# Function to preprocess the image
def preprocess_image(image_path):
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found.")
        return None

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read '{image_path}'. Check file path/integrity.")
        return None

    # Validate background color
    if not validate_background_color(image):
        print("Error: Invalid background color.")
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise and improve OCR accuracy
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding to preprocess the image
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Apply morphological operations to remove noise
    kernel = np.ones((1, 1), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    
    return morph

# Function to extract text using Tesseract OCR
def extract_text(image):
    # Use Tesseract to extract text
    text = pytesseract.image_to_string(image, lang='eng', config='--psm 6')
    return text

# Function to detect KTP components
def detect_ktp_components(image):
    # Use Tesseract to extract data with bounding boxes
    data = pytesseract.image_to_data(image, output_type=Output.DICT)
    n_boxes = len(data['text'])
    
    for i in range(n_boxes):
        if int(data['conf'][i]) > 60:  # Confidence threshold
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            image = cv2.putText(image, data['text'][i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    
    return image, data

# Function to validate the structure of KTP
def validate_ktp(text):
    required_fields = ["NIK", "Nama", "Tempat/Tgl Lahir", "Alamat", "RT/RW", "Kel/Desa", "Kecamatan", "Agama", "Status Perkawinan", "Pekerjaan", "Kewarganegaraan", "Berlaku Hingga"]
    
    valid = True
    for field in required_fields:
        if field not in text:
            valid = False
            break
    
    return valid

# Main function to process a single image
def process_image(image_path):
    preprocessed_image = preprocess_image(image_path)
    if preprocessed_image is None:
        return
    
    text = extract_text(preprocessed_image)
    
    print(f"Extracted Text from {image_path}:")
    print(text)
    
    is_valid = validate_ktp(text)
    if is_valid:
        print(f"KTP from {image_path} is valid.")
    else:
        print(f"KTP from {image_path} is invalid.")
    
    ktp_image, data = detect_ktp_components(preprocessed_image)
    output_image_path = f"output_{os.path.basename(image_path)}"
    cv2.imwrite(output_image_path, ktp_image)
    print(f"Detected KTP components saved to {output_image_path}")

# Main function to process all images in a directory
def process_all_images(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory_path, filename)
            process_image(image_path)

# Example usage
if __name__ == "__main__":
    directory_path = 'data'  # The directory containing the images
    process_all_images(directory_path)
