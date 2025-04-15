from PIL import Image
import torch

def generate_caption(model, processor, image, device):
    model.eval()
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = model.generate(pixel_values=inputs.pixel_values, max_length=50)
    return processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]