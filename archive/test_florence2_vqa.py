import os
import json
import logging
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
MODEL_ID = "microsoft/Florence-2-large"

def test_florence2():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch_dtype
    ).to(device)
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    image = Image.open('Sample.pdf_page_0.png').convert("RGB")
    
    prompt = """<VQA> Analyze this engineering drawing. I need you to identify what each detected number/dimension represents.
These numbers were found by OCR in the drawing:
1. "10"
2. "8"
3. "410"

Text labels found:
- "SECTION A-A"

For EACH number above, tell me:
- "feature": a short name like outer_diameter
- "value": copy the exact number from the OCR list
- "unit": write mm
- "type": one word from: linear, diameter
- "notes": which part or view it belongs to

Output ONLY valid JSON with this structure, nothing else:
{"columns":["id","feature","value","unit","type","notes"],"data":[]}"""
    
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
    
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        do_sample=False,
        num_beams=3
    )
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    result = processor.post_process_generation(generated_text, task="<VQA>", image_size=(image.width, image.height))
    
    with open('fl4_out.txt', 'w', encoding='utf-8') as f:
        f.write(str(result))
    print("DONE writing to fl4_out.txt")

if __name__ == "__main__":
    test_florence2()
