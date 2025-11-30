# SemEval 2026 Task 4: Narrative Similarity Baselines

This repository contains the baseline implementation for SemEval 2026 Task 4 (Track A).
It includes a modular pipeline for data loading, preprocessing, training, and evaluation of three baseline models.

## Pipeline Overview

1.  **Data Loading**: `data_loader.py` reads the JSONL dataset.
2.  **Preprocessing**: `preprocess.py` handles tokenization.
3.  **Models**:
    *   **Baseline A**: GPT-4o improved (Prompt-based, `model_baseline_A.py`).
    *   **Baseline B**: LSTM-based Siamese Network (`model_baseline_B.py`).
    *   **Baseline C**: BERT fine-tuned (`model_baseline_C.py`).
4.  **Training**: `train.py` trains Baseline B and C.
5.  **Evaluation**: `evaluate.py` computes accuracy and generates plots.

## Setup

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Set your OpenAI API key (for Baseline A):
    ```bash
    export OPENAI_API_KEY="your-key-here"
    ```

## Usage

### Training

To train Baseline B (LSTM):
```bash
python train.py --model B --epochs 5 --batch_size 8
```

To train Baseline C (BERT):
```bash
python train.py --model C --epochs 3 --batch_size 4
```

Checkpoints will be saved in `output/`.

### Evaluation

To evaluate Baseline A (GPT-4o):
```bash
python evaluate.py --model A
```

To evaluate Baseline B (LSTM):
```bash
python evaluate.py --model B --checkpoint output/model_B_epoch_5.pt
```

To evaluate Baseline C (BERT):
```bash
python evaluate.py --model C --checkpoint output/model_C_epoch_3.pt
```

Results (metrics and plots) will be saved in `results/`.

## Colab Notebook

A Google Colab notebook `baseline_pipeline_colab.ipynb` is provided to run the entire pipeline in the cloud.
