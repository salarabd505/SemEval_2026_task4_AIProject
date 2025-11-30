import argparse
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, logging as hf_logging

# Suppress tokenizer warnings
hf_logging.set_verbosity_error()

from data_loader import load_track_a
from model_baseline_A import GPTBaseline, ResponseEnum
from model_baseline_B import LSTMSimilarityModel
from model_baseline_C import BertSimilarityModel
from train import TrackADataset

RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True, parents=True)


import random

def evaluate(args):
    # Ensure reproducibility for splitting
    random.seed(42)
    
    raw_data = load_track_a(args.data_path)
    
    # Split data exactly like train.py
    random.shuffle(raw_data)
    split_idx = int(len(raw_data) * 0.8)
    train_data = raw_data[:split_idx]
    val_data = raw_data[split_idx:]
    
    if args.split == "train":
        data = train_data
        print(f"Evaluating on TRAINING set ({len(data)} examples)")
    elif args.split == "val":
        data = val_data
        print(f"Evaluating on VALIDATION set ({len(data)} examples)")
    else:
        data = raw_data
        print(f"Evaluating on FULL set ({len(data)} examples)")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = []
    ground_truth = []

    if args.model == "A":
        model = GPTBaseline()
        for i, item in enumerate(data):
            anchor = item["anchor"]
            choice_a = item["choice_a"]
            choice_b = item["choice_b"]
            label = item["label"]  # "A" or "B"
            
            try:
                pred = model.predict(anchor, choice_a, choice_b)
                predictions.append(pred)
                ground_truth.append(label)
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(data)}")
            except Exception as e:
                print(f"Error on example {i}: {e}")
                predictions.append("A") # Default fallback
                ground_truth.append(label)

    elif args.model == "B":
        vocab_path = Path("output/vocab.json")
        if not vocab_path.exists():
            raise FileNotFoundError("Vocab file not found. Run train.py --model B first.")
        
        with open(vocab_path, "r") as f:
            vocab = json.load(f)
            
        dataset = TrackADataset(data, model_type="B", vocab=vocab)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        
        model = LSTMSimilarityModel(vocab_size=len(vocab)).to(device)
        if args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint, map_location=device))
            print(f"Loaded checkpoint {args.checkpoint}")
        else:
            print("Warning: No checkpoint provided, using random weights.")
            
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                anchor = batch["anchor"].to(device)
                choice_a = batch["choice_a"].to(device)
                choice_b = batch["choice_b"].to(device)
                labels = batch["label"].cpu().numpy() # 0 or 1
                
                logits = model(anchor, choice_a, choice_b)
                preds = torch.argmax(logits, dim=1).cpu().numpy() # 0 or 1
                
                predictions.extend(["A" if p == 0 else "B" for p in preds])
                ground_truth.extend(["A" if l == 0 else "B" for l in labels])

    elif args.model == "C":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        dataset = TrackADataset(data, tokenizer=tokenizer, model_type="C")
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        
        model = BertSimilarityModel().to(device)
        if args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint, map_location=device))
            print(f"Loaded checkpoint {args.checkpoint}")
        else:
            print("Warning: No checkpoint provided, using random weights.")
            
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                input_ids_a = batch["input_ids_a"].to(device)
                mask_a = batch["attention_mask_a"].to(device)
                input_ids_b = batch["input_ids_b"].to(device)
                mask_b = batch["attention_mask_b"].to(device)
                labels = batch["label"].cpu().numpy()
                
                logits = model(input_ids_a, mask_a, input_ids_b, mask_b)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                
                predictions.extend(["A" if p == 0 else "B" for p in preds])
                ground_truth.extend(["A" if l == 0 else "B" for l in labels])

    # Calculate Accuracy
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    accuracy = correct / len(ground_truth) if ground_truth else 0
    print(f"Accuracy: {accuracy:.4f}")
    
    # Save metrics
    metrics = {"accuracy": accuracy, "model": args.model}
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    # Plot
    plt.figure(figsize=(6, 4))
    plt.bar([f"Baseline {args.model}"], [accuracy], color='skyblue')
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy")
    plt.title(f"Baseline {args.model} Performance")
    plt.savefig(PLOTS_DIR / f"accuracy_baseline_{args.model}.pdf")
    print(f"Plot saved to {PLOTS_DIR / f'accuracy_baseline_{args.model}.pdf'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["A", "B", "C"])
    parser.add_argument("--data_path", type=str, default="data/dev_track_a.jsonl")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint (for B/C)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "all"], help="Which split to evaluate on")
    args = parser.parse_args()
    
    evaluate(args)
