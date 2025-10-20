# HLNN Final Model – Hierarchical LSTM-CNN for Multi-Class News Classification

## Overview

This repository contains the code and resources for the Hierarchical LSTM-CNN News Classification Model (HLNN), a hybrid deep learning framework designed for interpretable multi-class news classification. HLNN captures both local syntactic features (via CNN) and global semantic dependencies (via LSTM), enhanced by attention-based topic modeling, prototype learning, and domain adaptation modules.

The framework has been evaluated on UCI News Aggregator, BBC News, and a self-curated hybrid dataset, demonstrating robust performance and cross-domain generalization.

## Repository Structure
```
HLNN_Code/
│
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── preprocessing.py           # Text cleaning, tokenization, and augmentation
├── model_hlnn.py              # HLNN model architecture (CNN + LSTM + Attention)
├── train.py                   # Training pipeline with early stopping and checkpointing
├── evaluate.py                # Evaluation metrics, confusion matrix, and attention visualizations
└── utils/
    └── metrics.py             # Custom metrics and explainability functions
```

## Environment Setup

**Python version:** 3.11  
**Required libraries:** listed in `requirements.txt`

```bash
pip install -r requirements.txt
```

**Key libraries include:**
- `tensorflow (2.x)` – Model implementation
- `numpy`, `pandas` – Data processing
- `scikit-learn` – Metrics and label encoding
- `matplotlib`, `seaborn` – Visualizations
- `spacy`, `nltk` – Text preprocessing and tokenization

## Dataset Overview

| Dataset | Samples | Classes | Description |
|---------|---------|---------|-------------|
| All the News | 101,111 | 7 | Full-text news articles from two major online archives |
| UCI News Aggregator | 422,420 | 4 | Headlines categorized by publishers |
| BBC News | 2,226 → ~20,000 (augmented) | 5 | Curated news articles, expanded using hybrid semantic augmentation |
| Hybrid Dataset | 44,646 | 9 | Combined UCI + BBC datasets with harmonized labels |

**Class categories (Hybrid Dataset):** Business, Technology, Education, Entertainment, Politics, Sports, Opinion, World, Science  
**Partitioning:** 80:20 train-validation stratified split; no data leakage or duplication across splits.

## Preprocessing and Augmentation

- Token-level normalization: lowercasing, punctuation removal, stopword removal, and lemmatization.
- NER filtering: removes named entities to prevent shortcuts in classification.
- Sequence padding/truncation: all sequences fixed at 300 tokens.
- Class encoding: labels converted to one-hot vectors.

**Hybrid Semantic Augmentation (BBC dataset):**
- Back-translation (English → French → English)
- Transformer-based paraphrasing

**Dataset fusion:** UCI + BBC harmonized to 9 classes, verified for lexical diversity and class balance.

## Model Architecture

**HLNN Backbone:**
- Embedding Layer: 50-dimensional trainable embeddings, optionally fused with pretrained embeddings (GloVe/FastText/BERT).
- CNN Path: Conv1D (64 filters, kernel size 5) → MaxPooling → Flatten
- LSTM Path: LSTM (80 units) → Flatten
- Concatenation: CNN + LSTM outputs
- Dense Layer: 64 units, ReLU
- Dropout: 0.4
- Output Layer: Softmax, 9 classes

**Advanced Modules:**
- Neural Variational Document Model (NVDM): End-to-end latent topic modeling
- Multi-Head Attention & HAN: Token- and sentence-level attention for interpretability
- Prototype Learning: Adaptive topic centroids
- Domain-Adversarial Module: Gradient Reversal Layer for cross-domain robustness

**Trainable parameters:** 70,017

## Training

**Hyperparameters (final configuration):**

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 16 |
| Epochs | 30 |
| Embedding Dimension | 50 |
| LSTM Units | 80 |
| CNN Filters | 64 |
| Kernel Size | 5 |
| Dropout Rate | 0.4 |
| Class Balancing | SMOTE |
| Topic Clusters (K) | 9 |
| Shuffle | True |

**Callbacks:** EarlyStopping and ModelCheckpoint implemented in `train.py`.  
**Usage:**
```bash
python train.py
```

## Evaluation

**Metrics computed on validation/test set (Hybrid Dataset):**
- Accuracy
- Precision, Recall, F1-Score (weighted)
- Confusion Matrix
- Attention-based explainability:
  - Visualizes token-level importance via multi-head attention.
  - Provides quantitative measures like attention entropy and qualitative human-centric validation.

**Usage:**
```bash
python evaluate.py
```

**Example Attention Visualization:**
```python
from evaluate import plot_attention
plot_attention(tokenizer, "Sample news text...", attention_weights)
```
Displays top-K tokens with the highest attention scores as a bar chart.

## Notes
- Ensure datasets are pre-cleaned CSVs with columns `text` and `label`.
- Hybrid semantic augmentation is currently a placeholder; custom augmentation scripts can be integrated.
- The model is fully compatible with TensorFlow 2.x and Python 3.11.