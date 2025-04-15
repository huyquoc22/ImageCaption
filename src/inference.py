from PIL import Image
import torch

def generate_caption(model, tokenizer, image, device="cuda", prefix=""):
    model.eval()
    model.to(device)

    max_text_len = 40
    prefix_encoding = tokenizer(prefix, padding='do_not_pad', truncation=True,
                                add_special_tokens=False, max_length=max_text_len)
    input_ids = [tokenizer.cls_token_id] + prefix_encoding["input_ids"]

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model({
            "image": image,
            "prefix": torch.tensor(input_ids).unsqueeze(0).to(device),
        })

    caption = tokenizer.decode(output["predictions"][0].tolist(), skip_special_tokens=True)
    return caption
