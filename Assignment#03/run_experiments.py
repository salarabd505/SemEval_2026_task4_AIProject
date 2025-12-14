import argparse
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np

from data_loader import load_track_a
from model_proposed import BiLSTMAttentionSimilarityModel
from utils import plot_training_history, build_vocab

# Set seed for reproducibility
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

OUTPUT_DIR = Path("results")
PLOTS_DIR = Path("plots")
OUTPUT_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

class TrackADataset(Dataset):
    def __init__(self, data, vocab, max_len=128):
        self.data = data
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        anchor = item["anchor"]
        choice_a = item["choice_a"]
        choice_b = item["choice_b"]
        
        # Label: 0 if A is closer, 1 if B is closer
        label = 0 if item["label"] == "A" else 1

        def text_to_ids(text):
            tokens = text.lower().split()[:self.max_len]
            return [self.vocab.get(t, self.vocab.get("<unk>", 0)) for t in tokens]

        anchor_ids = text_to_ids(anchor)
        a_ids = text_to_ids(choice_a)
        b_ids = text_to_ids(choice_b)
        
        def pad(ids):
            if len(ids) < self.max_len:
                return ids + [0] * (self.max_len - len(ids))
            return ids[:self.max_len]

        return {
            "anchor": torch.tensor(pad(anchor_ids), dtype=torch.long),
            "choice_a": torch.tensor(pad(a_ids), dtype=torch.long),
            "choice_b": torch.tensor(pad(b_ids), dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long)
        }

def train(args):
    # Load data
    print(f"Loading data from {args.data_path}...")
    try:
        raw_data = load_track_a(args.data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {args.data_path}")
        print("Please ensure the data file is present or update the path.")
        return

    # Split
    random.shuffle(raw_data)
    split_idx = int(len(raw_data) * 0.8)
    train_data = raw_data[:split_idx]
    val_data = raw_data[split_idx:]
    
    print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")

    # Build Vocab
    vocab = build_vocab(train_data)
    print(f"Vocab size: {len(vocab)}")
    
    # Save vocab
    with open(OUTPUT_DIR / "vocab.json", "w") as f:
        json.dump(vocab, f)

    # Datasets
    train_dataset = TrackADataset(train_data, vocab)
    val_dataset = TrackADataset(val_data, vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = BiLSTMAttentionSimilarityModel(vocab_size=len(vocab)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            anchor = batch["anchor"].to(device)
            choice_a = batch["choice_a"].to(device)
            choice_b = batch["choice_b"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            logits = model(anchor, choice_a, choice_b)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                anchor = batch["anchor"].to(device)
                choice_a = batch["choice_a"].to(device)
                choice_b = batch["choice_b"].to(device)
                labels = batch["label"].to(device)
                
                logits = model(anchor, choice_a, choice_b)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pt")
            print("Saved best model.")

    # Plot history
    plot_training_history(history, PLOTS_DIR / "training_curves.pdf")
    
    # Save final history
    with open(OUTPUT_DIR / "history.json", "w") as f:
        json.dump(history, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../data/dev_track_a.jsonl")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    
    train(args)
