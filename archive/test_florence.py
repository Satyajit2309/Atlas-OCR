import os
import cv2
import json
import logging
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import torch

# ──────────────────── LOGGING ────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("atlas-florence2-test")

# ──────────────────── CONFIG ────────────────────
MODEL_ID = "microsoft/Florence-2-large"

def test_florence2(image_path):
    """Run Florence-2 object detection/OCR on an image"""
    log.info(f"Loading {MODEL_ID}...")
    
    # We use CPU by default unless a supported GPU is found
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: {device}")
    
    try:
        # Load the model with appropriate precision
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, 
            trust_remote_code=True, 
            torch_dtype=torch_dtype
        ).to(device)
        
        processor = AutoProcessor.from_pretrained(
            MODEL_ID, 
            trust_remote_code=True
        )
        
        log.info("Model loaded successfully. Processing image...")
        
        # Load and convert image
        image = Image.open(image_path).convert("RGB")
        
        # We can ask Florence-2 to do <OCR> or <OD> (Object Detection)
        # Let's try finding dense OCR with regions to map everything out.
        prompt = "<OCR_WITH_REGION>"
        
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
        
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3
        )
        
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(
            generated_text, 
            task=prompt, 
            image_size=(image.width, image.height)
        )
        
        log.info(f"Florence-2 Raw Response (truncated): {str(parsed_answer)[:500]}...")
        
        # Example of saving results
        with open("florence2_test_results.json", "w") as f:
            json.dump(parsed_answer, f, indent=2)
            
        log.info("Saved full results to florence2_test_results.json")
        
    except Exception as e:
        log.error(f"Error executing Florence-2: {e}")

if __name__ == "__main__":
    test_img = "Sample.pdf_page_0.png"
    if os.path.exists(test_img):
        test_florence2(test_img)
    else:
        log.info(f"Please generate {test_img} first by running app.py processing pipeline.")
