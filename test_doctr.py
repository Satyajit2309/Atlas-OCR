"""Quick standalone test of the docTR pipeline on Sample.pdf"""
import os
os.environ["USE_TORCH"] = "1"

import json
import logging
import numpy as np
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("doctr-test")

def test():
    log.info("Loading docTR model...")
    model = ocr_predictor(
        det_arch='db_resnet50',
        reco_arch='parseq',
        pretrained=True,
        assume_straight_pages=False,
        straighten_pages=True,
        export_as_straight_boxes=True,
    )

    import torch
    if torch.cuda.is_available():
        model = model.cuda()
        log.info(f"Running on GPU: {torch.cuda.get_device_name(0)}")
    else:
        log.info("Running on CPU")

    log.info("Loading Sample.pdf...")
    doc = DocumentFile.from_pdf("Sample.pdf")
    log.info(f"Loaded {len(doc)} page(s)")

    log.info("Running OCR...")
    result = model(doc)
    json_output = result.export()

    log.info("=== ALL DETECTED TEXT ===")
    for page_idx, page in enumerate(json_output['pages']):
        log.info(f"--- Page {page_idx + 1} (dims: {page['dimensions']}) ---")
        for block in page['blocks']:
            for line in block['lines']:
                line_text = ' '.join([w['value'] for w in line['words']])
                line_conf = np.mean([w['confidence'] for w in line['words']])
                log.info(f"  [{line_conf:.3f}] \"{line_text}\"")

    # Save full JSON for inspection
    with open('doctr_sample_output.json', 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    log.info("Full JSON saved to doctr_sample_output.json")

if __name__ == "__main__":
    test()
