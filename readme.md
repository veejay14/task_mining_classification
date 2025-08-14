# **Task Classification with Multimodal Event Logs**

This project implements a full pipeline for **Task Mining / Step Name classification** from event logs.  
It supports **event name**, **categorical**, **numeric**, **text**, and **image** features, with flexible fusion into sequence models (**BiLSTM** or **Transformer**).  
The pipeline includes **data preparation, training with hyperparameter optimization, evaluation (with misclassification analysis), and inference**.

---

## **Features**
- **Sequence models**:
  - **BiLSTM** (bidirectional LSTM with stacked layers)
  - **Transformer Encoder**
- **Multimodal fusion**:
  - Event embeddings
  - Categorical features → embeddings
  - Numeric features → projection
  - Text features → SentenceTransformer embeddings
  - Image features → ResNet-50 embeddings
- Flexible fusion modes: `concat` or `gated`
- **Hyperparameter optimization** with Optuna
- **Evaluation**:
  - Plain accuracy, macro-F1, classification report
  - Selective evaluation with confidence threshold
  - Misclassification analysis (confusion patterns & per-class metrics)
- **Inference**:
  - Apply trained model on new raw event logs
  - Selective predictions with coverage metrics

---

## **Architecture Overview**

           ┌───────────────────┐
           │   Event Tokens     │
           └─────────┬─────────┘
                     │  Embedding
                     ▼
           ┌───────────────────┐
           │ Event Embeddings   │
           └─────────┬─────────┘
                     │
    ┌────────────────┼──────────────────┐
    │                │                  │
    ▼                ▼                  ▼
Categorical Embeddings Numeric Projection Text Embeddings
(from OrdinalEncoder) (from StandardScaler) (SentenceTransformer)
    │                │                  │
    └───────┬────────┴──────────┬───────┘
            │                   │
            ▼                   ▼
      Image Embeddings    (Optional Modalities)
       (ResNet-50)

            │
            ▼
   Concatenate / Gated Fusion
            │
            ▼
  ┌──────────────────────┐
  │  Sequence Encoder     │
  │  (BiLSTM / Transformer) 
  └───────────┬──────────┘
              │
              ▼
     Token-wise Classification
     (Step Name Prediction)
  
---

## **Project Structure**
├── config/
│ └── config.py # Global configuration (YAML/Dict)
├── data_preparation/
│ ├── data_pipeline.py # All feature engineering & encoding
│ ├── data_loading.py # EventLogDataBuilder (raw → preprocessed DF)
├── evaluation/
│ └── evaluator.py # Plain & selective eval + misclassification analysis
├── modelling/
│ └── models.py # BiLSTM & Transformer models with multimodal fusion
├── hyperparameter_tuning/
│ └── tuner.py # Optuna HPO routines
├── training_pipeline.py # End-to-end training script
├── inference.py # Inference on new data
└── README.md # This file


---

## **Setup**

### **1. Install dependencies**
```bash
pip install -r requirements.txt

---

## **Setup**

### **1. Install dependencies**
```bash
pip install -r requirements.txt

2. Prepare your data

Input CSV should contain:

Session column (e.g., session_id)

Timestamp column (sortable order)

Event name column (e.g., event_name)

Step name column (label for classification)

Optional categorical/numeric/text/image columns

Edit config/config.py or YAML file:

data:
  session_col: session_id
  timestamp_col: timestamp
  event_col: event_name
  label_col: step_name
  cat_features: ['action_type', 'target_app', 'file_extension']
  num_features: ['file_path_depth', 'is_sharepoint', 'event_name_length']
  text_features: ['event_name']        # Leave empty [] to disable
  image_path_col: image_path           # Null to disable
  batch_size: 32
  test_size: 0.2
  val_size: 0.25
  random_seed: 42

model:
  model_type: bilstm                   # or transformer
  fusion_mode: concat                  # or gated
  emb_event_dim: 48
  emb_cat_dim: 12
  num_proj_dim: 12
  txt_proj_dim: 32
  img_proj_dim: 32
  lstm_hidden_dim: 96
  lstm_num_layers: 2
  lstm_dropout: 0.4
  lstm_rnn_dropout: 0.2
  d_model: 128
  nhead: 4
  tf_num_layers: 2
  tf_ff_dim: 256
  tf_dropout: 0.3
  pad_idx: 0

train:
  lr: 1e-3
  weight_decay: 5e-4
  num_epochs: 50
  grad_clip: 1.0
  patience: 5
  ignore_index: -100
  precision_threshold: 0.95
  hpo:
    enabled: false
    metric: macro_f1

### Training

Run the training pipeline:
python training_pipeline.py
This will:

Load and preprocess the data

Split into train/val/test sets

Optionally run HPO

Train the selected model

Save the best checkpoint to trained_artifacts/best.pt

Save encoders/scalers to trained_artifacts/

Evaluation

During training, evaluation is done:

Plain evaluation (all valid tokens)

Selective evaluation (only predictions with confidence ≥ threshold)

Misclassification analysis: shows top confusions, per-class confusion, per-class metrics

Inference

To run the model on new event log data:

python inference.py

This will:

Load the trained checkpoint and artifacts

Preprocess the new CSV

Encode features (including text/image if configured)

Predict step names with confidence

Optionally abstain from low-confidence predictions

Save results to predictions.csv

Example Outputs

Plain Evaluation:

=== Test set (plain) ===
loss: 0.3052 | token-acc: 0.914 | macro-F1: 0.908
...
=== Misclassification analysis (plain test) ===
Top confusions (True → Predicted):
  StepA → StepB : 42  (12.7%)
  ...

Selective Evaluation (threshold=0.95):

Coverage: 0.83
loss: 0.2101 | token-acc: 0.953 | macro-F1: 0.950

Notes

If you enable text features, ensure sentence-transformers is installed.

If you enable image features, ensure torchvision and Pillow are installed.

The fusion mode (concat / gated) can significantly affect performance.

Selective evaluation helps measure precision at high confidence thresholds.

License

MIT License — feel free to use and adapt.


