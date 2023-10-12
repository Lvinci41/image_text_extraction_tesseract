import cv2
import os
import pytesseract
from PIL import Image
 
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
current_path = os.listdir(os.getcwd())
border = "********************"
 
for images in current_path:
    #convert to grayscale image
    img = cv2.imread(images)
    if img is None:
        continue

    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray=cv2.threshold(gray, 0,255,cv2.THRESH_BINARY| cv2.THRESH_OTSU)[1]
    gray=cv2.medianBlur(gray, 3)
        
    #memory usage with image i.e. adding image to memory
    filename = "{}".format(str(images))
    cv2.imwrite(filename, gray)
    text = pytesseract.image_to_string(Image.open(filename))
    os.remove(filename)
    #print(text)
    with open("results.txt", "a+") as file:
        file.write("%s\n%s\n%s\n%s\n\n" % (border, filename, border, text))

file.close()
