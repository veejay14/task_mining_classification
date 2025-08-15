# Task Classification with Multimodal Event Logs

A comprehensive deep learning pipeline for **sequence labeling** in event logs using multimodal features. This project predicts step names for each event in a session using advanced neural networks with support for text, images, categorical, and numeric features.

## 🎯 Problem Overview

In task mining and business process analysis, event logs record user interactions, actions, and system events over time. Each event belongs to a session (a continuous block of activity) and is associated with a step name that identifies the stage of a task.

**The Goal**: Given a sequence of events in a session, predict the correct step name for each event.

This is **sequence labeling** — the output is a label for every element in a sequence.

## 🚀 Key Challenges

- **Sequential Nature**: Events are temporally ordered; predictions must consider context
- **Multimodal Features**: 
  - Event names (discrete tokens, domain-specific)
  - Categorical features (app names, file extensions)
  - Numeric features (time deltas, path depth)
  - Text descriptions (free-form)
  - Images (screenshots, UI elements)
- **Class Imbalance**: Some step names appear rarely
- **Noisy Logs**: Inconsistent event names, similar steps
- **High Accuracy Requirements**: Prefer abstaining over wrong predictions

## 🏗️ Architecture Overview

```
    ┌─────────────────┐
    │   Event Logs    │
    └─────────┬───────┘
              │
    ┌─────────▼───────┐
    │ Feature Extract │ 
    │ & Encoding      │
    └─────────┬───────┘
              │
    ┌─────────▼───────┐
    │ Multimodal      │
    │ Fusion Layer    │
    └─────────┬───────┘
              │
    ┌─────────▼───────┐
    │ BiLSTM or       │
    │ Transformer     │
    └─────────┬───────┘
              │
    ┌─────────▼───────┐
    │ Token-wise      │
    │ Classification  │
    └─────────────────┘
```

## 📁 Project Structure

```
src/
├── config/
│   ├── config.py              # YAML configuration loader
│   └── config.yaml            # Training & model configuration
├── data_preparation/
│   ├── data_loading.py        # Raw CSV → processed DataFrame
│   └── data_pipeline.py       # Feature engineering & encoding
├── modelling/
│   └── models.py              # BiLSTM & Transformer architectures
├── evaluation/
│   └── evaluator.py           # Evaluation metrics & analysis
├── hyperparameter_tuning/
│   └── tuner.py               # K-fold cross-validation HPO
├── training_pipeline.py       # End-to-end training script
├── inference.py               # Model inference on new data
└── baseline_model.py          # Simple ML baseline
scripts/                       # Entry point scripts
data/                          # Input datasets
trained_artifacts/             # Model checkpoints & encoders
results/                       # Inference outputs
```

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- UV package manager (recommended) or pip

### Setup with UV (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd task_mining_classification

# Install dependencies
uv sync

# Install with optional multimodal features
uv sync --extra multimodal --extra dev
```

### Setup with pip

```bash
pip install -r requirements.txt.old

# For multimodal features
pip install sentence-transformers torchvision Pillow

# For development
pip install jupyter matplotlib seaborn
```

## 📊 Data Preparation

### Input Data Format

Your CSV should contain the following columns:

| Column | Description | Required |
|--------|-------------|----------|
| `session_id` | Session identifier for grouping events | ✅ |
| `timestamp` | Event timestamp (sortable) | ✅ |
| `event_name` | Primary event identifier | ✅ |
| `step_name` | Target classification label | ✅ |
| `action_type` | Derived categorical feature | ⚠️ Auto-generated |
| `target_app` | Application name | ⚠️ Auto-generated |
| `file_extension` | File extension if present | ⚠️ Auto-generated |
| Custom features | Additional categorical/numeric columns | ❌ Optional |

### Example Data

```csv
session_id,timestamp,event_name,step_name
1,2024-01-01 10:00:00,Write: Document.pdf,Document_Creation
1,2024-01-01 10:01:00,Read: Settings.ini,Configuration
1,2024-01-01 10:02:00,Action: Save File,Document_Creation
```

### Configuration

Edit `src/config/config.yaml` to match your data:

```yaml
data:
  data_csv: "data/your_dataset.csv"
  session_col: "session_id"
  timestamp_col: "timestamp"
  event_col: "event_name"
  label_col: "step_name"
  cat_features: ["action_type", "target_app", "file_extension"]
  num_features: ["file_path_depth", "elapsed_time"]
  text_features: []  # ["event_name"] to enable text features
  # image_path_col: "image_path"  # Uncomment to enable image features
  
  test_size: 0.10    # 10% for testing
  val_size: 0.10     # 10% for validation  
  batch_size: 32
  random_seed: 42

model:
  model_type: "transformer"  # "bilstm" or "transformer"
  fusion_mode: "concat"      # "concat" or "gated"
  
  # Embedding dimensions
  emb_event_dim: 64
  emb_cat_dim: 16
  num_proj_dim: 16
  
  # BiLSTM parameters
  lstm_hidden_dim: 128
  lstm_num_layers: 2
  lstm_dropout: 0.4
  
  # Transformer parameters
  d_model: 128
  nhead: 4
  tf_num_layers: 4
  tf_ff_dim: 256
  tf_dropout: 0.3

train:
  lr: 0.001
  num_epochs: 50
  patience: 2
  precision_threshold: 0.95
  
  hpo:
    enabled: false    # Enable k-fold cross-validation HPO
    k_folds: 5
    n_trials: 20
    epochs_per_fold: 10
```

## 🏋️ Training

### Basic Training

```bash
# Using UV
uv run python scripts/train.py

# Or directly
uv run python -m src.training_pipeline
```

### With Hyperparameter Optimization

1. Enable HPO in `src/config/config.yaml`:
   ```yaml
   train:
     hpo:
       enabled: true
       k_folds: 5
       n_trials: 20
   ```

2. Run training:
   ```bash
   uv run python scripts/train.py
   ```

The training pipeline will:
1. **Load and preprocess** the data with feature engineering
2. **Split by sessions** (80% train, 10% val, 10% test)  
3. **Build vocabularies** and fit encoders on training data only
4. **Optimize hyperparameters** (if enabled) using k-fold cross-validation
5. **Train the final model** on combined train+val data (when HPO enabled)
6. **Evaluate on test set** with both standard and selective metrics
7. **Save artifacts** to `trained_artifacts/`

### Training Output

```
[TRAINING] Using combined train+val data (64,349 samples) for final training

epoch 1 train loss: 1.2543
           test (for early stopping only) loss: 0.8901 | token-acc: 0.756

=== Test set (plain) ===
loss: 0.3052 | token-acc: 0.914 | macro-F1: 0.908

=== Test set (selective @ 0.95) ===  
Coverage: 0.832 | loss: 0.2101 | token-acc: 0.953 | macro-F1: 0.950

=== Misclassification Analysis ===
Top confusions (True → Predicted):
  Paperwork → Data_Entry : 125 (8.3%)
  Configuration → Paperwork : 89 (6.2%)
```

## 🔬 Fine-tuning

### Hyperparameter Search Space

The system automatically optimizes:

- **Embedding dimensions**: Event, categorical, numeric projections
- **Architecture parameters**: Hidden dimensions, layer counts, dropout rates
- **Learning parameters**: Learning rate, weight decay
- **Model-specific**: LSTM vs Transformer parameters

### K-fold Cross-Validation

When HPO is enabled, the system:
1. Uses k-fold CV on combined train+validation data (90%)
2. Evaluates each parameter combination across all folds
3. Selects best parameters based on average fold performance
4. Trains final model on full train+val data
5. Evaluates only once on reserved test set (10%)

This prevents overfitting and provides robust hyperparameter selection.

## 🔮 Inference

### Running Inference

1. Place your new event log data in `data/predictions.csv`
2. Ensure the same format as training data
3. Run inference:

```bash
# Using UV
uv run python scripts/inference.py

# Or directly  
uv run python -m src.inference
```

### Inference Process

The system will:
1. **Load trained model** and preprocessing artifacts
2. **Apply same feature engineering** as during training
3. **Encode sequences** using saved vocabularies and scalers
4. **Generate predictions** with confidence scores
5. **Apply selective prediction** (optional abstention)
6. **Save results** to `results/predictions.csv`

### Output Format

```csv
session_id,event_name,pred_step_name,pred_confidence,pred_step_id
1,Write: Document.pdf,Document_Creation,0.987,2
1,Read: Settings.ini,Configuration,0.923,1
1,Action: Save File,Document_Creation,0.991,2
```

### Selective Prediction

With `precision_threshold: 0.95`, the model only predicts when ≥95% confident:
- **Higher precision**: Fewer false positives
- **Lower coverage**: Some events remain unlabeled
- **Business value**: Send uncertain cases for manual review

```
Coverage: 0.832 | Precision: 0.953 | F1: 0.950
```

## 🧠 Model Architecture Deep Dive

### Stage 1: Feature Engineering

- **Event names** → vocabulary indices (PAD=0, UNK=1)
- **Categorical features** → OrdinalEncoder with PAD/UNK handling
- **Numeric features** → StandardScaler normalization
- **Text features** → SentenceTransformer embeddings (optional)
- **Image features** → ResNet-50 embeddings (optional)

### Stage 2: Multimodal Fusion

#### Concatenation Mode
```
fused = [event_emb; cat_embs; num_proj; text_proj; img_proj]
```

#### Gated Fusion Mode  
```
fused = base + α_text × text_proj + α_img × img_proj
where α weights are learned per modality
```

### Stage 3: Sequence Modeling

#### BiLSTM Option
- **Bidirectional LSTM** captures past and future context
- **Packed sequences** for efficient variable-length processing
- **Gradient clipping** prevents exploding gradients
- **Good for**: Smaller datasets, interpretable temporal modeling

#### Transformer Option
- **Self-attention** models all pairwise interactions
- **Positional encoding** preserves sequence order
- **Multi-head attention** captures different relationship types
- **Good for**: Large datasets, complex long-range dependencies

### Stage 4: Classification

- **Token-wise linear layer** maps encoder outputs to step name logits
- **CrossEntropyLoss** with ignore_index for padded positions
- **Confidence thresholding** enables selective prediction

## 📈 Evaluation Metrics

### Standard Metrics
- **Token Accuracy**: Percentage of correctly classified events
- **Macro F1-Score**: Balanced performance across all step names
- **Classification Report**: Per-class precision, recall, F1

### Selective Evaluation
- **Coverage**: Percentage of events with confident predictions
- **Selective Accuracy**: Accuracy only on confident predictions
- **Precision-Coverage Trade-off**: Balance abstention vs errors

### Misclassification Analysis
- **Top Confusions**: Most frequent error patterns
- **Per-class Analysis**: Which classes are hardest to predict
- **Confusion Heatmaps**: Visual error pattern analysis

## 🔧 Advanced Configuration

### Multimodal Features

Enable text features:
```yaml
data:
  text_features: ["event_name"]
model:
  txt_proj_dim: 32
```

Enable image features:
```yaml
data:
  image_path_col: "screenshot_path"
model:
  img_proj_dim: 32
```

### Model Architecture Selection

**BiLSTM** for:
- Smaller datasets (< 50K events)
- Clear temporal patterns
- Interpretability requirements
- Limited computational resources

**Transformer** for:
- Large datasets (> 100K events)
- Complex interaction patterns
- Long sequences
- When accuracy is paramount

## 🎯 Best Practices

### Data Quality
- **Consistent event naming**: Standardize event vocabularies
- **Temporal ordering**: Ensure accurate timestamps
- **Session boundaries**: Clear session start/end points
- **Balanced labels**: Address severe class imbalances

### Model Selection
- **Start with BiLSTM**: Faster training, good baseline
- **Scale to Transformer**: For complex patterns, large data
- **Enable multimodal**: When text/images add value
- **Use selective prediction**: For high-stakes applications

### Hyperparameter Optimization
- **Enable k-fold HPO**: For robust parameter selection
- **Monitor overfitting**: Watch train vs validation metrics
- **Early stopping**: Prevent overtraining
- **Gradient clipping**: Stabilize training

## 🔍 Troubleshooting

### Common Issues

**Memory errors during training:**
- Reduce `batch_size` in config
- Use gradient accumulation
- Enable mixed precision training

**Poor performance:**
- Check data quality and labeling consistency
- Enable hyperparameter optimization
- Try different fusion modes
- Add more training data

**Slow inference:**
- Reduce model complexity
- Use CPU for small batches
- Enable batch processing


