# Import the required modules
import pytesseract
from PIL import Image

# Set the path for the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Open the image using Pillow library
img = Image.open('image.png')

# Convert the image to grayscale
img = img.convert('L')

# Perform OCR using Tesseract
text = pytesseract.image_to_string(img)

# Print the extracted text
print(text)
