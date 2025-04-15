from pycocoevalcap.cider.cider import Cider
import torch
from tqdm import tqdm
import evaluate

def compute_cider(predictions, references):
    scorer = Cider()
    score, _ = scorer.compute_score({i: [ref] for i, ref in enumerate(references)},
                                    {i: [pred] for i, pred in enumerate(predictions)})
    return score

def evaluate_model(model, tokenizer, dataset, device="cuda"):
    model.to(device)
    model.eval()

    predictions = []
    references = []

    max_text_len = 40
    prefix = ""

    prefix_encoding = tokenizer(prefix, padding='do_not_pad', truncation=True,
                                add_special_tokens=False, max_length=max_text_len)
    input_ids = [tokenizer.cls_token_id] + prefix_encoding["input_ids"]

    for example in tqdm(dataset, desc="Evaluating"):
        image = example["image"].unsqueeze(0).to(device)
        gt_caption = example["text"]

        with torch.no_grad():
            output = model({
                "image": image,
                "prefix": torch.tensor(input_ids).unsqueeze(0).to(device),
            })

        generated_caption = tokenizer.decode(output["predictions"][0].tolist(), skip_special_tokens=True)
        predictions.append(generated_caption)
        references.append(gt_caption)

    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")

    bleu.add_batch(predictions=predictions, references=[[r] for r in references])
    rouge.add_batch(predictions=predictions, references=references)
    meteor.add_batch(predictions=predictions, references=references)

    bleu_result = bleu.compute()
    rouge_result = rouge.compute()
    meteor_result = meteor.compute()["meteor"]
    cider_result = compute_cider(predictions, references)

    result = (
        f"BLEU-4: {bleu_result['bleu']:.4f}\n"
        f"ROUGE-L: {rouge_result['rougeL']:.4f}\n"
        f"METEOR: {meteor_result:.4f}\n"
        f"CIDEr: {cider_result:.4f}\n"
    )

    return result
