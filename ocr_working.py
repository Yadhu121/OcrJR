"""
PaddleOCR Script - Working with PaddleOCR 2.7.0
Uses the correct .ocr() method
"""

import os

# Disable OneDNN before importing
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['PADDLE_INFERENCE_USE_MKLDNN'] = '0'
os.environ['FLAGS_use_onednn'] = '0'
os.environ['FLAGS_enable_pir_api'] = '0'
os.environ['FLAGS_enable_pir_in_executor'] = '0'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

from paddleocr import PaddleOCR

image_path = 'philips.jpeg'

print("Initializing PaddleOCR...")
print("(CPU-only mode to avoid crashes)\n")

# Initialize OCR
ocr = PaddleOCR(lang='en')

print(f"Processing: {image_path}")
print("Please wait...\n")

try:
    # Use .ocr() method (correct for version 2.7.0)
    result = ocr.ocr(image_path, cls=True)
    
    print("=" * 70)
    print("RECOGNIZED TEXT:")
    print("=" * 70)
    
    all_text = []
    
    if result and result[0]:
        for idx, line in enumerate(result[0], 1):
            # Format: [bbox, (text, confidence)]
            bbox = line[0]
            text = line[1][0]
            confidence = line[1][1]
            
            print(f"{idx}. {text}")
            print(f"    Confidence: {confidence:.1%}")
            all_text.append(text)
    
    print("=" * 70)
    
    if all_text:
        # Save to file
        with open('output.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(all_text))
        
        print(f"\n✓ SUCCESS! Recognized {len(all_text)} lines of text")
        print(f"✓ Saved to: output.txt\n")
        
        print("Full text:")
        print("-" * 70)
        for line in all_text:
            print(line)
        print("-" * 70)
    else:
        print("\n⚠ No text was recognized in the image")
        
except Exception as e:
    print(f"\n❌ ERROR: {e}\n")
    
    import traceback
    traceback.print_exc()
