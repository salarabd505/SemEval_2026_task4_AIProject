import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, logging as hf_logging

# Suppress tokenizer warnings
hf_logging.set_verbosity_error()

from data_loader import load_track_a
from preprocess import tokenize
from model_baseline_B import LSTMSimilarityModel
from model_baseline_C import BertSimilarityModel

# Set seed for reproducibility
random.seed(42)
torch.manual_seed(42)

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


class TrackADataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer=None, model_type="B", max_len=128, vocab=None):
        self.data = data
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.max_len = max_len
        self.vocab = vocab  # Dict[str, int] for LSTM

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        anchor = item["anchor"]
        choice_a = item["choice_a"]
        choice_b = item["choice_b"]
        # Label: 0 if A is closer (text_a_is_closer=True), 1 if B is closer
        # The JSON has "label" as "A" or "B" in my data_loader, 
        # BUT wait, the original file has "text_a_is_closer": boolean.
        # My data_loader.py converts it? Let's check data_loader.py.
        # Ah, data_loader.py expects "label" key but the file has "text_a_is_closer".
        # I need to fix data_loader.py or handle it here.
        # Let's check data_loader.py content again.
        # It says: `required = {"anchor", "choice_a", "choice_b", "label"}`
        # But the file has `anchor_text`, `text_a`, `text_b`, `text_a_is_closer`.
        # I MUST FIX data_loader.py first! 
        # I will fix data_loader.py in a separate step. 
        # For now, let's assume data_loader returns standardized keys: anchor, choice_a, choice_b, label (0 or 1).
        
        label = 0 if item["label"] == "A" else 1

        if self.model_type == "B":
            # LSTM: Tokenize and map to integers using vocab
            def text_to_ids(text):
                tokens = text.lower().split()[:self.max_len]
                return [self.vocab.get(t, self.vocab.get("<unk>", 0)) for t in tokens]

            anchor_ids = text_to_ids(anchor)
            a_ids = text_to_ids(choice_a)
            b_ids = text_to_ids(choice_b)
            
            # Padding handled in collate_fn or here. Let's do simple padding here.
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

        elif self.model_type == "C":
            # BERT: Tokenize pairs
            # Pair 1: Anchor + Choice A
            enc_a = self.tokenizer(
                anchor,
                choice_a,
                truncation=True,
                max_length=self.max_len,
                padding="max_length",
                return_tensors="pt"
            )
            # Pair 2: Anchor + Choice B
            enc_b = self.tokenizer(
                anchor,
                choice_b,
                truncation=True,
                max_length=self.max_len,
                padding="max_length",
                return_tensors="pt"
            )
            
            return {
                "input_ids_a": enc_a["input_ids"].squeeze(0),
                "attention_mask_a": enc_a["attention_mask"].squeeze(0),
                "input_ids_b": enc_b["input_ids"].squeeze(0),
                "attention_mask_b": enc_b["attention_mask"].squeeze(0),
                "label": torch.tensor(label, dtype=torch.long)
            }
        return {}


def build_vocab(data, min_freq=1):
    counts = {}
    for item in data:
        for key in ["anchor", "choice_a", "choice_b"]:
            for word in item[key].lower().split():
                counts[word] = counts.get(word, 0) + 1
    
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, count in counts.items():
        if count >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def train(args):
    # Load data
    # Note: I need to fix data_loader.py to handle the actual file format!
    # I will assume it is fixed or I will patch it.
    # For now, I'll use load_track_a and assume it works or I'll fix it next.
    raw_data = load_track_a(args.data_path)
    
    # Split
    random.shuffle(raw_data)
    split_idx = int(len(raw_data) * 0.8)
    train_data = raw_data[:split_idx]
    val_data = raw_data[split_idx:]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.model == "A":
        print("Baseline A (GPT-4o) does not require training. Use evaluate.py directly.")
        return

    elif args.model == "B":
        vocab = build_vocab(train_data)
        print(f"Vocab size: {len(vocab)}")
        dataset = TrackADataset(train_data, model_type="B", vocab=vocab)
        val_dataset = TrackADataset(val_data, model_type="B", vocab=vocab)
        
        model = LSTMSimilarityModel(vocab_size=len(vocab)).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        # Save vocab
        with open(OUTPUT_DIR / "vocab.json", "w") as f:
            json.dump(vocab, f)

    elif args.model == "C":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        dataset = TrackADataset(train_data, tokenizer=tokenizer, model_type="C")
        val_dataset = TrackADataset(val_data, tokenizer=tokenizer, model_type="C")
        
        model = BertSimilarityModel().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            labels = batch["label"].to(device)
            
            if args.model == "B":
                anchor = batch["anchor"].to(device)
                choice_a = batch["choice_a"].to(device)
                choice_b = batch["choice_b"].to(device)
                logits = model(anchor, choice_a, choice_b)
            else: # C
                input_ids_a = batch["input_ids_a"].to(device)
                mask_a = batch["attention_mask_a"].to(device)
                input_ids_b = batch["input_ids_b"].to(device)
                mask_b = batch["attention_mask_b"].to(device)
                logits = model(input_ids_a, mask_a, input_ids_b, mask_b)

            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss/len(dataloader):.4f}")
        
        # Save checkpoint
        torch.save(model.state_dict(), OUTPUT_DIR / f"model_{args.model}_epoch_{epoch+1}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["A", "B", "C"])
    parser.add_argument("--data_path", type=str, default="data/dev_track_a.jsonl")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    
    train(args)
