import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from pathlib import Path

def plot_training_history(history, save_path):
    """
    Plots training and validation loss and accuracy.
    Args:
        history: dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: path to save the plot (PDF)
    """
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()
    print(f"Saved training history plot to {save_path}")

def plot_confusion_matrix(y_true, y_pred, save_path, labels=['A', 'B']):
    """
    Plots confusion matrix.
    Args:
        y_true: list of true labels
        y_pred: list of predicted labels
        save_path: path to save the plot (PDF)
        labels: list of label names
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()
    print(f"Saved confusion matrix to {save_path}")

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
