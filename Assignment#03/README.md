# Assignment 3: Proposed Solution (Siamese Bi-LSTM with Attention)

This folder contains the proposed solution for SemEval 2026 Task 4 (Track A).
The model is a Siamese Bi-LSTM with a Self-Attention mechanism to aggregate hidden states, replacing the simple max-pooling of the baseline.

## Files
- `model_proposed.py`: Implementation of `BiLSTMAttentionSimilarityModel`.
- `run_experiments.py`: Training script with validation and plotting.
- `evaluate_results.py`: Evaluation script generating metrics and confusion matrix.
- `data_loader.py`: Data loading utility.
- `utils.py`: Helper functions for plotting.
- `requirements.txt`: Python dependencies.

## How to Run on Google Colab

1. **Upload Files**:
   - Create a folder named `Assignment 3` in your Colab runtime (or just upload files to the root).
   - Upload all `.py` files and `requirements.txt` from this folder.
   - Upload your dataset file (e.g., `dev_track_a.jsonl`) to a `data` folder or the root.

   **Option B: Google Drive (Recommended)**
   1. Upload the `Assignment 3` folder to your Google Drive.
   2. Mount Drive in Colab:
      ```python
      from google.colab import drive
      drive.mount('/content/drive')
      ```
   3. Navigate to the folder:
      ```bash
      %cd /content/drive/MyDrive/path/to/Assignment\ 3
      ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Model**:
   ```bash
   # If data is in the root directory:
   python run_experiments.py --data_path dev_track_a.jsonl --epochs 10 --batch_size 16
   ```
   - This will create `results/` (saved model, vocab) and `plots/` (training curves).

4. **Evaluate the Model**:
   ```bash
   python evaluate_results.py --data_path dev_track_a.jsonl
   ```
   - This will generate `plots/confusion_matrix.pdf` and print accuracy.

## Model Architecture
- **Embedding**: Learnable embeddings.
- **Encoder**: Bidirectional LSTM (2 layers).
- **Aggregation**: Self-Attention layer (weighted sum of hidden states).
- **Similarity**: Cosine similarity between Anchor-A and Anchor-B.
- **Output**: Logits for choice A vs B.
