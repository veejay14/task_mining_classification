"""Simple machine learning baseline for event log classification.

This module provides a lightweight RandomForest baseline for comparison
with the more sophisticated deep learning models. It demonstrates basic
feature engineering and evaluation approaches.

The baseline performs the following steps:
1. Loads raw CSV event logs
2. Engineers simple features from event_name and session timing
3. Encodes features into model-ready matrices (one-hot + numeric)
4. Trains RandomForest classifier with cross-validation
5. Evaluates on validation and test sets
6. Reports high-confidence predictions with threshold analysis

This serves as a sanity check and performance baseline for comparing
more complex neural network approaches.

Usage:
    python src/baseline_model.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import logging


# -----------------------------------------------------------------------------
# Feature engineering helpers (kept intentionally small & obvious)
# -----------------------------------------------------------------------------

def extract_action_type(event_name: str):
    """
    Pull the word before a colon, e.g. 'Open:' -> 'Open'.
    We also strip '*' (seen in some logs) and normalize casing.
    """
    match = re.match(r"\s*([\w\*]+)\s*:", str(event_name), re.IGNORECASE)
    if match:
        return match.group(1).replace("*", "").capitalize()
    return None


def extract_target_app(event_name: str):
    """
    Capture the substring after 'in ' (like 'Open in SharePoint' -> 'SharePoint').
    It's a rough heuristic but very effective for many event strings.
    """
    match = re.search(r"in ([\w\.\-\\ ]+)", str(event_name), re.IGNORECASE)
    return match.group(1).strip() if match else None


def extract_file_extension(event_name: str):
    """Find a file extension token like '.pdf' or '.xlsx' (2–5 chars)."""
    match = re.search(r"\.([a-zA-Z0-9]{2,5})\b", str(event_name))
    return match.group(1).lower() if match else None


def extract_first_number(event_name: str):
    """Grab the first standalone number, useful for versions or counts."""
    match = re.search(r"\b(\d+)\b", str(event_name))
    return match.group(1) if match else None


def extract_file_path_depth(event_name: str):
    """
    Estimate Windows-style path depth by counting backslashes, e.g.
    'C:\\Users\\me\\Desktop\\file.txt' -> depth 3.
    """
    match = re.search(r"([a-zA-Z]:\\[^ ]+)", str(event_name))
    if match:
        path = match.group(1)
        return path.count("\\")
    return None


# -----------------------------------------------------------------------------
# Modular, OO components
# -----------------------------------------------------------------------------

class FeatureEngineer:
    """
    Turns raw event rows into a richer DataFrame with helpful columns.

    Why this exists:
    - Keeps regex and time logic out of your modeling code.
    - Makes unit testing painless (you can feed in tiny DataFrames and check outputs).
    """

    def __init__(self, event_col: str = "event_name", time_col: str = "timestamp", session_col: str = "session_id"):
        self.event_col = event_col
        self.time_col = time_col
        self.session_col = session_col

    def add_event_name_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        col = self.event_col

        df["action_type"] = df[col].apply(extract_action_type)
        df["target_app"] = df[col].apply(extract_target_app)
        df["file_extension"] = df[col].apply(extract_file_extension)
        df["file_path_depth"] = df[col].apply(extract_file_path_depth)
        df["is_sharepoint"] = df[col].str.contains("sharepoint", case=False, na=False)
        df["event_name_length"] = df[col].astype(str).str.len()
        df["first_number"] = df[col].apply(extract_first_number)
        return df

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds:
        - elapsed_time: seconds since the previous event in the same session
        - session_duration: total time (seconds) from first to last event in the session
        """
        df = df.copy()
        tcol, scol = self.time_col, self.session_col

        # Robust parsing (supports mixed formats)
        df[tcol] = pd.to_datetime(df[tcol], format="mixed")

        # Sort so diffs are meaningful
        df.sort_values(by=[scol, tcol], inplace=True)

        # Time gap within session
        df["elapsed_time"] = df.groupby(scol)[tcol].diff().dt.total_seconds().fillna(0)

        # Session duration: (max - min) per session
        session_duration = df.groupby(scol)[tcol].agg(lambda x: (x.max() - x.min()).total_seconds())
        df["session_duration"] = df[scol].map(session_duration)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full engineering pass: event-name features + time features.
        """
        df = self.add_event_name_features(df)
        df = self.add_time_features(df)
        return df


@dataclass
class PreprocessConfig:
    """
    Configuration for preprocessing. Tweak here; no need to touch code below.
    """
    categorical_features: Tuple[str, ...] = ("action_type", "event_name", "target_app", "group_id", "file_extension")
    numeric_features: Tuple[str, ...] = ("file_path_depth", "event_name_length", "elapsed_time", "session_duration")
    boolean_features: Tuple[str, ...] = ("is_sharepoint",)


class Preprocessor:
    """
    Converts engineered DataFrame into (X, y) ready for modeling.

    What it does:
    - One-hot encodes categorical columns (including 'event_name' if you want that granularity).
    - Ensures numeric columns are numeric (coerce errors to NaN).
    - Converts booleans to 0/1.
    - Label-encodes the target (`step_name` -> `target`).
    """

    def __init__(self, config: PreprocessConfig | None = None, target_col: str = "step_name"):
        self.config = config or PreprocessConfig()
        self.target_col = target_col
        self.label_encoder = LabelEncoder()  # persisted for reporting

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
        df = df.copy()

        # Encode target
        df["target"] = self.label_encoder.fit_transform(df[self.target_col])
        y = df["target"]

        # Booleans -> ints
        for col in self.config.boolean_features:
            if col in df.columns:
                df[col] = df[col].astype(int)

        # Numerics -> clean numeric dtype
        for col in self.config.numeric_features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # One-hot categoricals
        cat_cols_present = [c for c in self.config.categorical_features if c in df.columns]
        X_cat = pd.get_dummies(df[cat_cols_present], dummy_na=True) if cat_cols_present else pd.DataFrame(index=df.index)

        # Combine blocks
        X_num_bool_cols = [c for c in (list(self.config.numeric_features) + list(self.config.boolean_features)) if c in df.columns]
        X_num_bool = df[X_num_bool_cols] if X_num_bool_cols else pd.DataFrame(index=df.index)

        X = pd.concat([X_cat, X_num_bool], axis=1)
        return X, y, df  # return engineered df too (sometimes handy)

    @property
    def target_names_(self) -> np.ndarray:
        return self.label_encoder.classes_


@dataclass
class TrainConfig:
    """
    Training & evaluation knobs. Adjust once, reuse everywhere.
    """
    test_size: float = 0.20
    val_size_of_remaining: float = 0.25  # 0.25 of 0.80 -> 0.20 overall for val
    random_state: int = 42
    rf_estimators: int = 100
    confidence_threshold: float = 0.95


class BaselineModel:
    """
    A tiny wrapper that trains a RandomForest on your engineered/encoded data,
    prints friendly metrics, and also reports a high-confidence slice.

    Why a wrapper?
    - Keeps I/O, preprocessing, training, and evaluation clearly separated.
    - Prevents "leaking" mixing of concerns in your notebook or scripts.
    """

    def __init__(self, config: TrainConfig | None = None, preprocessor: Preprocessor | None = None):
        self.config = config or TrainConfig()
        self.preprocessor = preprocessor or Preprocessor()
        self.model = RandomForestClassifier(
            n_estimators=self.config.rf_estimators,
            random_state=self.config.random_state
        )

    # ------------------------ public API ------------------------

    def fit_eval(self, X: pd.DataFrame, y: pd.Series, *, target_names: np.ndarray | None = None):
        """
        Splits data into train/val/test, trains, and prints metrics.
        Follows your original split proportions exactly.
        """
        # 1) test split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )
        # 2) validation split from the remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=self.config.val_size_of_remaining,
            random_state=self.config.random_state,
            stratify=y_temp,
        )

        # Train
        self.model.fit(X_train, y_train)

        # Validation metrics
        y_val_pred = self.model.predict(X_val)
        print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
        if target_names is not None:
            print("Validation Classification Report:\n",
                  classification_report(y_val, y_val_pred, target_names=target_names))
        else:
            print(classification_report(y_val, y_val_pred))

        # Test metrics
        y_test_pred = self.model.predict(X_test)
        print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
        if target_names is not None:
            print("Test Classification Report:\n",
                  classification_report(y_test, y_test_pred, target_names=target_names))
        else:
            print(classification_report(y_test, y_test_pred))

        # High-confidence slice on validation
        self._report_high_confidence_slice(X_val, y_val, target_names)

    def _report_high_confidence_slice(self, X_val: pd.DataFrame, y_val: pd.Series, target_names: np.ndarray | None):
        """
        Prints metrics for predictions with probability >= confidence_threshold.
        This helps you see how well the model does when it's very sure.
        """
        proba = self.model.predict_proba(X_val)
        preds = np.argmax(proba, axis=1)
        max_p = np.max(proba, axis=1)

        # None for uncertain predictions; class index for confident ones
        y_pred_confident = np.array([p if m >= self.config.confidence_threshold else None for p, m in zip(preds, max_p)],
                                    dtype=object)

        # mask of confident predictions
        mask = pd.Series(y_pred_confident, dtype="object").notna().to_numpy()
        pct = mask.mean() * 100
        print(f"Proportion of predictions with ≥{self.config.confidence_threshold:.0%} confidence: {pct:.2f}%")

        if mask.sum() > 0:
            y_conf_true = y_val[mask]
            y_conf_pred = y_pred_confident[mask].astype(int)
            print("Accuracy on high-confidence predictions:", accuracy_score(y_conf_true, y_conf_pred))
            if target_names is not None:
                print("High-Confidence Classification Report:\n",
                      classification_report(y_conf_true, y_conf_pred, target_names=target_names))
            else:
                print(classification_report(y_conf_true, y_conf_pred))
        else:
            print("No predictions meet the confidence threshold. Try lowering it or improving the model.")


# -----------------------------------------------------------------------------
# Script entry point (kept tiny on purpose)
# -----------------------------------------------------------------------------

def main():
    # 1) Load data exactly like your original script
    data = pd.read_csv(
        "code_challenge_dataset.csv",
        encoding="latin1",
        on_bad_lines="skip",
        header=0,
        names=["timestamp", "session_id", "event_name", "group_id", "step_name"],
    )
    logging.info("Data Loaded")
    # 2) Engineer features
    fe = FeatureEngineer()
    engineered = fe.transform(data)
    logging.info("Feature Engineering Done")

    # 3) Preprocess to (X, y)
    pre = Preprocessor()
    X, y, engineered_df = pre.transform(engineered)
    logging.info("Preprocessing - Done!")

    # 4) Train & evaluate baseline
    logging.info("training and evaluation started")
    trainer = BaselineModel()
    trainer.fit_eval(X, y, target_names=pre.target_names_)


if __name__ == "__main__":
    main()
