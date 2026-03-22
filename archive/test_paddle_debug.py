from paddleocr import PaddleOCR
import cv2
import json

def test_paddle():
    reader = PaddleOCR(use_textline_orientation=True, lang='en')
    img = cv2.imread('Sample.pdf_page_0.png')
    result = reader.ocr(img, cls=True)
    
    with open('paddle_debug.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print("DONE writing to paddle_debug.json")

if __name__ == "__main__":
    test_paddle()
