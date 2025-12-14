import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

from data_loader import load_track_a
from model_proposed import BiLSTMAttentionSimilarityModel
from utils import plot_confusion_matrix

OUTPUT_DIR = Path("results")
PLOTS_DIR = Path("plots")
OUTPUT_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Vocab
    vocab_path = OUTPUT_DIR / "vocab.json"
    if not vocab_path.exists():
        print("Error: vocab.json not found. Run training first.")
        return
    
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    print(f"Loaded vocab size: {len(vocab)}")

    # Load Model
    model_path = OUTPUT_DIR / "best_model.pt"
    if not model_path.exists():
        print("Error: best_model.pt not found. Run training first.")
        return

    model = BiLSTMAttentionSimilarityModel(vocab_size=len(vocab)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Loaded best model.")

    # Load Data
    print(f"Loading data from {args.data_path}...")
    data = load_track_a(args.data_path)
    
    # If evaluating on same file as training (and split was random), 
    # ideally we should use the same split. 
    # For simplicity here, we just evaluate on the provided file.
    # In a real scenario, we'd have a separate test file.
    
    # Preprocess
    max_len = 128
    
    def text_to_ids(text):
        tokens = text.lower().split()[:max_len]
        return [vocab.get(t, vocab.get("<unk>", 0)) for t in tokens]
    
    def pad(ids):
        if len(ids) < max_len:
            return ids + [0] * (max_len - len(ids))
        return ids[:max_len]

    all_preds = []
    all_labels = []

    print("Evaluating...")
    with torch.no_grad():
        for item in data:
            anchor = torch.tensor([pad(text_to_ids(item["anchor"]))], dtype=torch.long).to(device)
            choice_a = torch.tensor([pad(text_to_ids(item["choice_a"]))], dtype=torch.long).to(device)
            choice_b = torch.tensor([pad(text_to_ids(item["choice_b"]))], dtype=torch.long).to(device)
            
            label = 0 if item["label"] == "A" else 1
            
            logits = model(anchor, choice_a, choice_b)
            pred = torch.argmax(logits, dim=1).item()
            
            all_preds.append(pred)
            all_labels.append(label)

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    print(f"Accuracy: {acc:.4f}")
    
    report = classification_report(all_labels, all_preds, target_names=['A', 'B'])
    print("Classification Report:")
    print(report)
    
    # Save metrics
    with open(OUTPUT_DIR / "evaluation_metrics.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(report)

    # Plot Confusion Matrix
    plot_confusion_matrix(all_labels, all_preds, PLOTS_DIR / "confusion_matrix.pdf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../data/dev_track_a.jsonl")
    args = parser.parse_args()
    
    evaluate(args)
