# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project implements a multimodal task mining and classification pipeline for event logs. It predicts step names from sequences of events using either BiLSTM or Transformer models with support for:

- Event name embeddings
- Categorical features (action_type, target_app, file_extension, group_id)
- Numerical features (file_path_depth, elapsed_time)
- Optional text features (using SentenceTransformer embeddings)  
- Optional image features (using ResNet-50 embeddings)

The pipeline includes data preparation, hyperparameter optimization with Optuna, training with early stopping, evaluation with misclassification analysis, and inference on new data.

## Core Commands

### Setup with UV
```bash
uv sync
```
Install all dependencies including PyTorch, scikit-learn, Optuna, and optional multimodal libraries using UV package manager.

```bash
uv sync --extra multimodal --extra dev
```
Install with optional dependencies for text/image features and development tools.

### Training
```bash
uv run python scripts/train.py
```
Or directly:
```bash
uv run python -m src.training_pipeline
```
Runs the full training pipeline including data loading, preprocessing, model training, and evaluation. Saves trained model and artifacts to `trained_artifacts/`.

### Inference
```bash
uv run python scripts/inference.py
```
Or directly:
```bash
uv run python -m src.inference
```
Applies trained model to new event log data from `data/predictions.csv` and saves predictions to `results/predictions.csv`.

### Data Exploration
The Jupyter notebook `data_exploration.ipynb` contains exploratory data analysis of the event log dataset including sequence patterns and feature distributions.

## Project Structure

```
src/
├── config/
│   ├── config.py          # YAML loader utility
│   └── config.yaml        # Central configuration
├── data_preparation/
│   ├── data_loading.py    # EventLogDataBuilder class
│   └── data_pipeline.py   # Feature engineering and datasets
├── modelling/
│   └── models.py          # BiLSTM and Transformer models
├── evaluation/
│   └── evaluator.py       # Evaluation utilities
├── hyperparameter_tuning/
│   └── tuner.py           # Optuna HPO routines
├── training_pipeline.py   # Main training script
├── inference.py           # Inference script
└── baseline_model.py      # Simple ML baseline
scripts/                   # Entry point scripts
data/                      # Dataset files
trained_artifacts/         # Model checkpoints and encoders
```

## Architecture

### Data Pipeline (`src/data_preparation/`)
- `data_loading.py`: EventLogDataBuilder class for raw CSV processing 
- `data_pipeline.py`: Feature engineering, encoding, sequence generation, and dataset classes

### Models (`src/modelling/models.py`)
- `MultiFeatureBiLSTMTagger`: Bidirectional LSTM with attention and multimodal fusion
- `MultiFeatureTransformerTagger`: Transformer encoder with positional encoding and fusion

Both models support:
- Concatenation or gated fusion of modalities
- Token-wise classification for step name prediction
- Flexible feature dimensions via configuration

### Configuration (`src/config/`)
- `config.yaml`: Central configuration for data paths, model hyperparameters, and training settings
- `config.py`: YAML loader utility

Key configuration sections:
- `data`: Input/output paths, feature columns, train/val/test splits
- `model`: Architecture choice (bilstm/transformer), embedding dimensions, fusion mode
- `train`: Learning rate, epochs, early stopping, hyperparameter optimization settings

### Training Pipeline
The training script handles:
1. Data loading and session-based train/val/test splitting (80/10/10)
2. Vocabulary building and encoder fitting on training data only
3. Optional k-fold cross-validation hyperparameter optimization with Optuna
4. Final model training using train+val data (when HPO enabled) with gradient clipping and early stopping
5. Evaluation with both plain accuracy and selective prediction (confidence thresholding)
6. Misclassification analysis showing confusion patterns

When hyperparameter optimization is enabled, the pipeline uses k-fold cross-validation on the combined train+validation data (90%) to find optimal hyperparameters, then trains the final model on this combined dataset. The test set (10%) remains completely unseen until final evaluation.

### Evaluation (`src/evaluation/evaluator.py`)
- Plain evaluation: Standard accuracy/F1 on all predictions
- Selective evaluation: Accuracy/F1 only on high-confidence predictions with coverage metrics
- Misclassification analysis: Confusion matrices and per-class error patterns

### Hyperparameter Tuning (`src/hyperparameter_tuning/tuner.py`)
Optuna-based optimization supporting:
- Model architecture parameters (hidden dimensions, layers, dropout)
- Learning rate and weight decay
- Configurable metrics (validation loss or macro F1)
- Trial pruning and timeout controls

## Key Implementation Details

### Sequence Processing
- Events are grouped by session_id and sorted by timestamp
- Sequences are padded dynamically in batches
- Labels use -100 for padding tokens (ignored in loss computation)

### Multimodal Fusion
- Features are projected to common dimensions then concatenated or gated
- Text features use sentence-transformers (all-MiniLM-L6-v2 by default)
- Image features use ResNet-50 pretrained embeddings
- Fusion modes: "concat" (simple concatenation) or "gated" (learned attention)

### Model Checkpointing
Trained artifacts saved to `trained_artifacts/`:
- `best.pt`: Best model checkpoint
- `event_vocab.json`: Event name vocabulary
- `label_encoder.pkl`: Step name label encoder  
- `cat_encoder.pkl`: Categorical feature encoder
- `num_scaler.pkl`: Numerical feature scaler

### Data Format
Input CSV should contain:
- `session_id`: Session identifier for grouping events
- `timestamp`: For temporal ordering within sessions  
- `event_name`: Primary event identifier
- `step_name`: Target classification label
- Optional categorical/numerical feature columns as configured

The dataset contains 71,831 events across 4,248 sessions with imbalanced step name distribution requiring careful evaluation approaches.