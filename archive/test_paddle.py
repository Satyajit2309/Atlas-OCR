from paddleocr import PaddleOCR
import cv2
import traceback

try:
    reader = PaddleOCR(use_textline_orientation=True, lang='en')
    img = cv2.imread('Sample.pdf_page_0.png')
    res = reader.ocr(img)
    print("SUCCESS using .ocr()")
except Exception as e:
    print("FAILED .ocr():")
    traceback.print_exc()

try:
    reader = PaddleOCR(use_textline_orientation=True, lang='en')
    img = cv2.imread('Sample.pdf_page_0.png')
    res = reader(img)
    print("SUCCESS using __call__()")
except Exception as e:
    print("FAILED __call__():")
    traceback.print_exc()
