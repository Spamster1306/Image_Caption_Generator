import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

def process_image_from_bytes(image_bytes):
    # Load image from bytes and convert to RGB format
    raw_image = Image.open(image_path).convert('RGB')
    
    # Initialize Blip processor and model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # Process image for unconditional captioning
    inputs = processor(raw_image, return_tensors="pt")
    
    # Generate caption
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    return caption

# Example usage with user-provided image file content as bytes
image_path = input("Enter the path to the image file: ")
unconditional_caption = process_image_from_bytes(image_path)
print(unconditional_caption)
