"""
Benchmark docTR detection sensitivity on Sample.pdf.
Try lower box_thresh values to catch more text.
"""
import os
os.environ['USE_TORCH'] = '1'

import json
import logging
import numpy as np
import cv2
import fitz  # PyMuPDF
from doctr.io import DocumentFile
from doctr.models import ocr_predictor, detection_predictor, recognition_predictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("bench")

def test_with_params():
    log.info("=== Testing docTR with tuned parameters ===")

    # Step 1: Convert PDF to high-DPI image manually first
    log.info("Converting Sample.pdf to 600 DPI image...")
    doc_pdf = fitz.open("Sample.pdf")
    page = doc_pdf[0]
    pix = page.get_pixmap(dpi=600)  # Higher DPI = more detail for small text
    img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
    else:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite("Sample_600dpi.png", img_bgr)
    doc_pdf.close()
    log.info(f"  Image size: {img_bgr.shape[1]}x{img_bgr.shape[0]}")

    # Step 2: Load with tuned model
    log.info("Loading docTR model with lower detection threshold...")

    # Use the predictor but with customized detection model
    det_model = detection_predictor(
        arch='db_resnet50',
        pretrained=True,
        assume_straight_pages=False,
        batch_size=1,
    )
    reco_model = recognition_predictor(
        arch='parseq',
        pretrained=True,
        batch_size=32,
    )

    model = ocr_predictor(
        det_arch='db_resnet50',
        reco_arch='parseq',
        pretrained=True,
        assume_straight_pages=False,
        straighten_pages=True,
        export_as_straight_boxes=True,
        detect_orientation=True,
    )
    # Lower the detection threshold to catch more text
    model.det_predictor.model.postprocessor.bin_thresh = 0.1
    model.det_predictor.model.postprocessor.box_thresh = 0.1

    import torch
    if torch.cuda.is_available():
        model = model.cuda()
        log.info(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Step 3: Run on the high-DPI image
    log.info("Running OCR on 600 DPI image...")
    doc = DocumentFile.from_images("Sample_600dpi.png")
    result = model(doc)
    json_output = result.export()

    log.info("=== ALL DETECTED TEXT (tuned) ===")
    for page in json_output['pages']:
        for block in page['blocks']:
            for line in block['lines']:
                for word in line['words']:
                    log.info(f"  [{word['confidence']:.3f}] obj={word['objectness_score']:.3f} \"{word['value']}\" orient={word.get('crop_orientation', {}).get('value', '?')}")

    with open('doctr_sample_tuned.json', 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    log.info("Saved to doctr_sample_tuned.json")

if __name__ == "__main__":
    test_with_params()
