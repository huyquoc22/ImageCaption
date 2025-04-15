from datasets import load_dataset, Image
from torch.utils.data import Dataset
import os
import torch
from transformers import BertTokenizer
from config import jsonl_file, root

def load_facad_dataset(split_ratio=(0.8, 0.1, 0.1), seed=42):

    dataset = load_dataset("json", data_files=jsonl_file)["train"]
    dataset = dataset.map(lambda x: {"image": os.path.join(root, x["image"])})
    dataset = dataset.cast_column("image", Image(decode=True))

    train_val_split = dataset.train_test_split(test_size=(1.0 - split_ratio[0]), seed=seed)
    train_dataset = train_val_split["train"]

    val_test_split = train_val_split["test"].train_test_split(
        test_size=split_ratio[2] / (split_ratio[1] + split_ratio[2]),
        seed=seed
    )
    val_dataset = val_test_split["train"]
    test_dataset = val_test_split["test"]

    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset
    }

class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, tokenizer=None, image_transform=None):
        self.dataset = dataset
        self.tokenizer = tokenizer or BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        self.image_transform = image_transform 

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        text = item["text"]
        prefix = ""

        max_text_len = 40
        prefix_encoding = self.tokenizer(prefix, padding='do_not_pad', truncation=True,
                                         add_special_tokens=False, max_length=max_text_len)
        target_encoding = self.tokenizer(text, padding='do_not_pad', truncation=True,
                                         add_special_tokens=False, max_length=max_text_len)

        need_predict = [0] * len(prefix_encoding['input_ids']) + [1] * len(target_encoding['input_ids'])
        payload = prefix_encoding['input_ids'] + target_encoding['input_ids']

        if len(payload) > max_text_len:
            payload = payload[-(max_text_len - 2):]
            need_predict = need_predict[-(max_text_len - 2):]

        input_ids = [self.tokenizer.cls_token_id] + payload + [self.tokenizer.sep_token_id]
        need_predict = [0] + need_predict + [1]

        data = {
            "caption_tokens": torch.tensor(input_ids),
            "need_predict": torch.tensor(need_predict),
            "image": image,
            "caption": {},
            "iteration": 0,
        }

        data = self.image_transform(data)
        data["attention_mask"] = data["caption_tokens"].ne(0).long()

        return data
