import argparse
import sys

import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification

def predict(sentence):
    model_dir = "./results"
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    label_map = {1: "积极", 0: "消极"}
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        pred_id = probs.argmax(dim=-1).item()

    print(pred_id, probs.squeeze().tolist())
    return label_map[pred_id], probs.squeeze().tolist()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "text",
            nargs="+")
        args = parser.parse_args()
        test_sentence = " ".join(args.text)
    else:
        test_sentence = "不错，下次还考虑入住。交通也方便，在餐厅吃的也不错。"
    label, prob = predict(test_sentence)
    print(f"Input   : {test_sentence}")
    print(f"Predicted label : {label} | Probabilities : {prob}")
