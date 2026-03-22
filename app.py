"""
Atlas-OCR: Local Engineering Drawing Dimension Extractor
=========================================================
Uses docTR (Document Text Recognition by Mindee) for high-accuracy OCR
on engineering drawings, including rotated/vertical text and symbols.

Pipeline:
  1. PDF → high-res (600 DPI) rasterisation via PyMuPDF
  2. docTR ocr_predictor (DBNet + PARSeq) on GPU, tuned thresholds
  3. Smart dimension parsing with regex (Ø, R, ±, °, CB, M, etc.)
  4. Noise filtering + deduplication
  5. Export to XLSX/CSV
"""

import os
os.environ["USE_TORCH"] = "1"

from flask import Flask, render_template, request, jsonify, send_file
import json
import re
import cv2
import numpy as np
import pandas as pd
import fitz  # PyMuPDF
import tempfile
import logging
from datetime import datetime
from werkzeug.utils import secure_filename

from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# ──────────────────── CONFIG ────────────────────

RENDER_DPI = 600            # High DPI so thin dimension text is readable
MIN_CONFIDENCE = 0.40       # Skip very low-confidence junk detections
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'}

# ──────────────────── LOGGING ────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("atlas-ocr")

# ──────────────────── FLASK ────────────────────

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB

# ──────────────────── LAZY GLOBALS ────────────────────

_doctr_model = None


def get_doctr_model():
    """Lazy-init docTR OCR predictor with tuned detection thresholds."""
    global _doctr_model
    if _doctr_model is None:
        log.info("Initializing docTR OCR predictor (first request may be slow)...")
        _doctr_model = ocr_predictor(
            det_arch='db_resnet50',
            reco_arch='parseq',
            pretrained=True,
            assume_straight_pages=False,
            straighten_pages=True,
            export_as_straight_boxes=True,
            detect_orientation=True,
        )
        # Lower detection thresholds — engineering drawings have small, thin text
        _doctr_model.det_predictor.model.postprocessor.bin_thresh = 0.1
        _doctr_model.det_predictor.model.postprocessor.box_thresh = 0.1

        import torch
        if torch.cuda.is_available():
            _doctr_model = _doctr_model.cuda()
            log.info(f"docTR loaded on GPU: {torch.cuda.get_device_name(0)}")
        else:
            log.info("docTR loaded on CPU")
    return _doctr_model


# ──────────────────── UTILITIES ────────────────────

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def pdf_to_high_res_images(pdf_path, dpi=RENDER_DPI):
    """Convert each PDF page to a high-res PNG for OCR."""
    doc = fitz.open(pdf_path)
    img_paths = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=dpi)
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        else:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_path = pdf_path + f"_page_{i}.png"
        cv2.imwrite(img_path, img_bgr)
        img_paths.append(img_path)
        log.info(f"  Page {i+1}: {img_bgr.shape[1]}x{img_bgr.shape[0]} px")
    doc.close()
    return img_paths


# ──────────────────── DIMENSION PARSING ────────────────────

# Words that are labels/annotations, NOT dimensions
NOISE_WORDS = {
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'N', 'O',
    'P', 'Q', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    '-', '+', '|', '/', '\\', '=', ':', ';', '(', ')', '[', ']',
    'a', 'b', 'c', 'd', 'e', 'o', 'l', 't',
    'VIEW', 'ISOMETRIC', 'SECTION', 'DETAIL', 'SCALE', 'FRONT',
    'TOP', 'SIDE', 'BOTTOM', 'RIGHT', 'LEFT', 'MATERIAL', 'DATE',
    'DRAWN', 'CHECKED', 'APPROVED', 'TITLE', 'SHEET', 'REV',
    'PROJECTION', 'TOLERANCES', 'FINISH', 'UNLESS', 'OTHERWISE',
    'SPECIFIED', 'DIMENSIONS', 'ARE', 'IN', 'MM', 'DO', 'NOT',
    'PART', 'NAME', 'QTY', 'DWG', 'NO', 'THE', 'ALL',
    'D.', 'A.', 'B.', 'N.',
}


def normalize_ocr_text(text):
    """
    Normalize docTR OCR output to fix common misreadings of engineering symbols.
    docTR reads: Ø → '$' or leading '0', ↓ → 'OL' or 'l'
    """
    text = text.strip()
    if not text:
        return text

    # docTR reads Ø as $ — replace $ followed by a number with Ø
    text = re.sub(r'\$\s*(\d)', r'Ø\1', text)
    # Standard normalization
    text = text.replace('⌀', 'Ø').replace('ø', 'Ø').replace('Φ', 'Ø').replace('φ', 'Ø')

    # docTR reads Ø as leading 0 before multi-digit numbers
    # e.g. "0133" → "Ø133" → we extract 133 as value
    # "030" → "Ø30", "06.5" → "Ø6.5"
    # Only apply when the number after the leading 0 is 2+ digits (to avoid "0" → "Ø")
    text = re.sub(r'^0(\d{2,}\.?\d*)$', r'Ø\1', text)

    # Remove stray spaces within numbers: "41 0" → "410"
    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)

    return text


def classify_dimension(text):
    """Classify a single OCR text into an engineering dimension."""
    text = normalize_ocr_text(text)
    if not text:
        return None

    # ── Diameter ──
    m = re.match(r'[ØÖ]\s*(\d+\.?\d*)', text)
    if m:
        return m.group(1), 'mm', 'diameter', f'diameter_{m.group(1)}'

    # ── Radius (but not words like "REV") ──
    m = re.match(r'R(\d+\.?\d*)$', text)
    if m:
        return m.group(1), 'mm', 'radius', f'radius_{m.group(1)}'

    # ── Thread (M8, M10x1.5) ──
    m = re.match(r'M(\d+\.?\d*)\s*[xX×]\s*(\d+\.?\d*)', text)
    if m:
        val = f'{m.group(1)}x{m.group(2)}'
        return val, 'mm', 'thread', f'thread_M{val}'
    m = re.match(r'M(\d+\.?\d*)$', text)
    if m:
        return m.group(1), 'mm', 'thread', f'thread_M{m.group(1)}'

    # ── Angle (45°) ──
    m = re.match(r'(\d+\.?\d*)\s*[°˚]', text)
    if m:
        return m.group(1), 'deg', 'angle', f'angle_{m.group(1)}'

    # ── Chamfer (2x45°) ──
    m = re.match(r'(\d+\.?\d*)\s*[xX×]\s*(\d+\.?\d*)\s*[°˚]?', text)
    if m:
        val = f'{m.group(1)}x{m.group(2)}'
        return val, 'mm/deg', 'chamfer', f'chamfer_{val}'

    # ── Tolerance (25±0.1) ──
    m = re.match(r'(\d+\.?\d*)\s*[±]\s*(\d+\.?\d*)', text)
    if m:
        return f'{m.group(1)}±{m.group(2)}', 'mm', 'tolerance', f'dim_{m.group(1)}_tol'

    # ── Plain number (2+ digits or decimal) ──
    m = re.match(r'^(\d{2,}\.?\d*)$', text.strip())
    if m:
        return m.group(1), 'mm', 'linear', f'dim_{m.group(1)}'

    # ── Single digit only if >= 1 (skip 0 noise) ──
    m = re.match(r'^(\d)$', text.strip())
    if m and m.group(1) != '0':
        return m.group(1), 'mm', 'linear', f'dim_{m.group(1)}'

    return None


def parse_compound_text(text):
    """Parse compound dimension annotations from docTR line text."""
    results = []
    text = text.strip()
    if not text:
        return results

    normalized = normalize_ocr_text(text)

    # ── "N HOLES – ØX" ──
    m = re.match(r'(\d+)\s*HOLES?\s*[-–—]?\s*[ØÖ]?\s*(\d+\.?\d*)', normalized, re.IGNORECASE)
    if m:
        results.append((m.group(1), 'count', 'hole_count', f'holes_{m.group(1)}'))
        results.append((m.group(2), 'mm', 'hole_diameter', f'hole_diameter_{m.group(2)}'))
        return results

    # ── "CB Ø X ↓ Y" ──
    m = re.match(r'C\.?B\.?\s*[ØÖ$]?\s*(?:OL|OI|l)?\s*(\d+\.?\d*)\s*(?:[↓⬇]|OL|OI|l)?\s*(\d+\.?\d*)?',
                 normalized, re.IGNORECASE)
    if m:
        results.append((m.group(1), 'mm', 'counterbore_dia', f'counterbore_dia_{m.group(1)}'))
        if m.group(2):
            results.append((m.group(2), 'mm', 'counterbore_depth', f'counterbore_depth_{m.group(2)}'))
        return results

    # ── "ØX ↓ Y" ──
    m = re.match(r'[ØÖ]\s*(\d+\.?\d*)\s*[↓⬇]\s*(\d+\.?\d*)', normalized)
    if m:
        results.append((m.group(1), 'mm', 'diameter', f'diameter_{m.group(1)}'))
        results.append((m.group(2), 'mm', 'depth', f'depth_{m.group(2)}'))
        return results

    # ── "CSK ØX" ──
    m = re.match(r'C\.?S\.?K\.?\s*[ØÖ]?\s*(\d+\.?\d*)\s*[°˚]?', normalized, re.IGNORECASE)
    if m:
        results.append((m.group(1), 'deg', 'countersink', f'countersink_{m.group(1)}'))
        return results

    # ── "6 HOLES" alone ──
    m = re.match(r'(\d+)\s*HOLES?', normalized, re.IGNORECASE)
    if m:
        results.append((m.group(1), 'count', 'hole_count', f'holes_{m.group(1)}'))
        return results

    # ── "↓5" or "DEPTH 10" ──
    m = re.match(r'[↓⬇]\s*(\d+\.?\d*)', normalized)
    if m:
        results.append((m.group(1), 'mm', 'depth', f'depth_{m.group(1)}'))
        return results
    m = re.match(r'DEPTH\s*(\d+\.?\d*)', normalized, re.IGNORECASE)
    if m:
        results.append((m.group(1), 'mm', 'depth', f'depth_{m.group(1)}'))
        return results

    # Fallback: single classification
    single = classify_dimension(normalized)
    if single:
        results.append(single)

    return results


# ──────────────────── MAIN PROCESSING ────────────────────

def process_drawing(filepath, filename):
    """Main processing pipeline using docTR."""
    cleanup_paths = []
    try:
        # ─── Step 1: Convert to high-res images ───
        log.info("Step 1: Converting to high-resolution images...")
        ext = filename.lower().rsplit('.', 1)[-1]

        if ext == 'pdf':
            img_paths = pdf_to_high_res_images(filepath, dpi=RENDER_DPI)
            cleanup_paths.extend(img_paths)
        else:
            img_paths = [filepath]

        log.info(f"  {len(img_paths)} page(s) to process")

        # ─── Step 2: Run OCR on each page ───
        log.info("Step 2: Running docTR OCR (GPU-accelerated)...")
        model = get_doctr_model()

        all_words = []
        for page_idx, img_path in enumerate(img_paths):
            page_no = page_idx + 1
            log.info(f"  OCR on page {page_no}...")

            doc = DocumentFile.from_images(img_path)
            result = model(doc)
            json_output = result.export()

            page = json_output['pages'][0]
            page_h = page['dimensions'][0]
            page_w = page['dimensions'][1]

            for block in page['blocks']:
                for line in block['lines']:
                    # Get each word individually (better than line-level for dims)
                    for word in line['words']:
                        conf = float(word['confidence'])
                        obj_score = float(word.get('objectness_score', 0))
                        val = word['value'].strip()

                        # Skip noise
                        if conf < MIN_CONFIDENCE:
                            continue
                        if val in NOISE_WORDS:
                            continue
                        if not val:
                            continue

                        wg = word['geometry']
                        wx_min = wg[0][0] * page_w
                        wy_min = wg[0][1] * page_h
                        wx_max = wg[1][0] * page_w
                        wy_max = wg[1][1] * page_h

                        all_words.append({
                            'text': val,
                            'confidence': conf,
                            'cx': (wx_min + wx_max) / 2,
                            'cy': (wy_min + wy_max) / 2,
                            'page': page_no,
                            'bbox': [wx_min, wy_min, wx_max, wy_max],
                        })

                    # Also try the combined line text for compound expressions
                    line_text = ' '.join([w['value'] for w in line['words']])
                    line_conf = np.mean([w['confidence'] for w in line['words']]) if line['words'] else 0
                    if line_conf >= MIN_CONFIDENCE and len(line['words']) > 1:
                        geo = line['geometry']
                        all_words.append({
                            'text': line_text,
                            'confidence': float(line_conf),
                            'cx': (geo[0][0] + geo[1][0]) / 2 * page_w,
                            'cy': (geo[0][1] + geo[1][1]) / 2 * page_h,
                            'page': page_no,
                            'bbox': [geo[0][0]*page_w, geo[0][1]*page_h,
                                     geo[1][0]*page_w, geo[1][1]*page_h],
                        })

        log.info(f"  Found {len(all_words)} candidate text regions")

        # Log all detected text for debugging
        for w in all_words:
            log.info(f"    [{w['confidence']:.2f}] p{w['page']}: \"{w['text']}\"")

        # ─── Step 3: Parse dimensions ───
        log.info("Step 3: Parsing dimensions...")
        all_data = []
        seen_values = set()
        idx = 1

        for w in all_words:
            text = w['text'].strip()
            if not text:
                continue

            parsed_dims = parse_compound_text(text)

            for (value, unit, dim_type, feature) in parsed_dims:
                # Deduplicate by (page, value, type)
                key = (w['page'], value, dim_type)
                if key in seen_values:
                    continue
                seen_values.add(key)

                all_data.append({
                    'id': str(idx),
                    'feature': feature,
                    'value': value,
                    'unit': unit,
                    'type': dim_type,
                    'confidence': f"{w['confidence']:.2f}",
                    'notes': f"page {w['page']}",
                })
                idx += 1

        log.info(f"  Parsed {len(all_data)} dimension entries")

        # ─── Step 4: Build result ───
        if not all_data:
            return None, "No dimensions could be extracted from the drawing.", None, None

        standard_columns = ['id', 'feature', 'value', 'unit', 'type', 'confidence', 'notes']

        result_data = {
            'columns': standard_columns,
            'data': all_data
        }

        # ─── Step 5: Export ───
        temp_dir = tempfile.gettempdir()
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        excel_name = f'dimensions_{ts}.xlsx'
        csv_name = f'dimensions_{ts}.csv'
        excel_path = os.path.join(temp_dir, excel_name)
        csv_path = os.path.join(temp_dir, csv_name)

        df = pd.DataFrame(all_data)
        existing_cols = [c for c in standard_columns if c in df.columns]
        df = df[existing_cols]

        df.to_excel(excel_path, index=False, engine='openpyxl')
        df.to_csv(csv_path, index=False)

        log.info(f"Exported {len(all_data)} rows to {excel_name}")

        return result_data, None, excel_name, csv_name

    except Exception as e:
        log.exception("Error in process_drawing")
        return None, str(e), None, None

    finally:
        for p in cleanup_paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except:
                pass


# ──────────────────── FLASK ROUTES ────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Supported: PDF, PNG, JPG, JPEG, BMP, TIF'}), 400

    temp_dir = tempfile.gettempdir()
    filename = secure_filename(file.filename)
    filepath = os.path.join(temp_dir, filename)

    try:
        file.save(filepath)
        result_data, error, excel_name, csv_name = process_drawing(filepath, filename)

        if error:
            return jsonify({'error': error}), 500

        return jsonify({
            'preview': result_data,
            'excel_file': excel_name,
            'csv_file': csv_name
        })

    except Exception as e:
        log.exception("Error processing file")
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

    finally:
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass


@app.route('/download/<filename>')
def download(filename):
    temp_dir = tempfile.gettempdir()
    safe_name = secure_filename(filename)
    path = os.path.join(temp_dir, safe_name)

    if not os.path.exists(path):
        return jsonify({'error': 'File not found'}), 404

    mimetype = 'text/csv' if safe_name.endswith('.csv') else \
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'

    return send_file(path, as_attachment=True, download_name=safe_name, mimetype=mimetype)


if __name__ == '__main__':
    log.info("=" * 60)
    log.info("    Atlas-OCR: Local Dimension Extractor")
    log.info("    Powered by docTR (Mindee)")
    log.info("=" * 60)
    log.info(f"  OCR Engine : docTR (DBNet + PARSeq)")
    log.info(f"  Render DPI : {RENDER_DPI}")
    log.info(f"  Min Conf   : {MIN_CONFIDENCE}")
    log.info("=" * 60)
    app.run(debug=True, port=5000, use_reloader=False)
