import torch
import random
from config import *
from datasets import load_facad_dataset, ImageCaptioningDataset
from evaluate import evaluate_model
from inference import generate_caption
from model import get_model
from transformers import BertTokenizer
from train import get_image_transform
from common import Config

def main():
    datasets = load_facad_dataset()
    test_raw = datasets["test"]

    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME, do_lower_case=True)
    cfg = {
        'crop_region_extend_in_datatransform': 4,
        'data_normalize': 'clip',
        'train_crop_size': 224,
        'input_small_scale': 0.8,
        'no_color_jitter': True,
        'no_flip': True,
        'no_aspect_dist': True,
        'interpolation': 'bicubic',
        'min_size_range32': [160, 224],
        'patch_size': 16,
        'train_transform': 'vitp',
    }
    cfg = Config(cfg, {})
    image_transform = get_image_transform(cfg)

    test_dataset = ImageCaptioningDataset(test_raw, tokenizer, image_transform)

    model = get_model(tokenizer, param={})
    model_path = f"{save_path}/last_model_caption_FA.pth"
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print("Evaluating model...")
    results = evaluate_model(model, tokenizer, test_dataset, device)
    print("Evaluation Metrics:\n", results)

    print("\nGenerating sample captions...\n")
    indices = random.sample(range(len(test_dataset)), 5)
    for i in indices:
        example = test_dataset[i]
        image = example["image"]
        actual = test_raw[i]["text"]
        caption = generate_caption(model, tokenizer, image, device)
        print(f"Image {i + 1}:")
        print(f"  Actual:    {actual}")
        print(f"  Predicted: {caption}")
        print()

if __name__ == "__main__":
    main()
