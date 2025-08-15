"""
Data loading & feature engineering for task-mining event logs.
"""

from __future__ import annotations
from typing import Optional, List, Dict
import re
import pandas as pd
import numpy as np


class EventFeatureExtractor:
    """
    String-based feature extraction from the raw 'event_name' column.
    Adds: action_type, target_app, file_extension, file_path_depth,
          is_sharepoint, event_name_length, first_number.
    """

    def __init__(self, event_col: str = "event_name"):
        self.event_col = event_col

    @staticmethod
    def extract_action_type(event_name: str) -> Optional[str]:
        """
        Matches any word (optionally containing '*') before a colon.
        Normalizes by stripping '*' and capitalizing.
        Examples:
          "Write: AVPageView ..." -> "Write"
          "Action*: IBBR - ..."   -> "Action"
        """
        if not isinstance(event_name, str):
            return None
        m = re.match(r"\s*([\w\*]+)\s*:", event_name, re.IGNORECASE)
        if m:
            # Capitalize first letter, but keep '*' if present
            word = m.group(1)
            return word[0].upper() + word[1:]
        return None

    @staticmethod
    def extract_target_app(event_name: str) -> Optional[str]:
        """
        Extracts app/site after 'in ' (e.g., 'in Acrobat Reader', 'in ibbr-global.dhl').
        """
        if not isinstance(event_name, str):
            return None
        m = re.search(r"in ([\w\.\-\\ ]+)", event_name, re.IGNORECASE)
        return m.group(1).strip() if m else None

    @staticmethod
    def extract_file_extension(event_name: str) -> Optional[str]:
        """
        Finds file extension tokens like '.pdf', '.xlsx' (2-5 alphanumeric chars).
        """
        if not isinstance(event_name, str):
            return None
        m = re.search(r"\.([a-zA-Z0-9]{2,5})\b", event_name)
        return m.group(1).lower() if m else None

    @staticmethod
    def extract_first_number(event_name: str) -> Optional[int]:
        """
        Returns the first integer present in the event name, else None.
        """
        if not isinstance(event_name, str):
            return None
        m = re.search(r"\b(\d+)\b", event_name)
        return int(m.group(1)) if m else None

    @staticmethod
    def extract_file_path_depth(event_name: str) -> Optional[int]:
        """
        Detects a Windows-style path like 'C:\\Users\\...\\file.ext'
        and returns the folder depth (count of backslashes).
        """
        if not isinstance(event_name, str):
            return None
        m = re.search(r"([a-zA-Z]:\\[^ ]+)", event_name)
        if not m:
            return None
        path = m.group(1)
        return path.count("\\")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract event based features for the given dataframe
        """
        col = self.event_col
        if col not in df.columns:
            raise KeyError(f"EventFeatureExtractor: '{col}' not found in DataFrame.")

        out = df.copy()
        out["action_type"] = out[col].apply(self.extract_action_type)
        out["target_app"] = out[col].apply(self.extract_target_app)
        out["file_extension"] = out[col].apply(self.extract_file_extension)
        out["file_path_depth"] = out[col].apply(self.extract_file_path_depth)
        out["is_sharepoint"] = out[col].astype(str).str.contains("sharepoint", case=False, regex=False)
        out["event_name_length"] = out[col].astype(str).str.len().astype("Int64")
        out["first_number"] = out[col].apply(self.extract_first_number)

        return out


class TimeFeatureEngineer:
    """
    Builds time-based features using session groupings:
      - elapsed_time: per-event delta seconds within a session (first = 0)
      - session_duration: (max_ts - min_ts) seconds per session
    """

    def __init__(self, timestamp_col: str = "timestamp", session_col: str = "session_id"):
        self.timestamp_col = timestamp_col
        self.session_col = session_col

    def _ensure_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[self.timestamp_col] = pd.to_datetime(out[self.timestamp_col], format="mixed", errors="coerce")
        return out

    def add_elapsed_time(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self._ensure_datetime(df)
        out = out.sort_values(by=[self.session_col, self.timestamp_col])
        deltas = out.groupby(self.session_col)[self.timestamp_col].diff().dt.total_seconds()
        out["elapsed_time"] = deltas.fillna(0.0).astype(float)
        return out

    def add_session_duration(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self._ensure_datetime(df)
        grp = out.groupby(self.session_col)[self.timestamp_col]
        duration = grp.transform(lambda x: (x.max() - x.min()).total_seconds() if x.notna().all() else np.nan)
        out["session_duration"] = duration.astype(float)
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self.add_elapsed_time(df)
        out = self.add_session_duration(out)
        return out


class EventLogDataBuilder:
    """
    Orchestrates:
      1) CSV load (with robust schema handling)
      2) String feature extraction from event_name
      3) Time features (elapsed_time, session_duration)
      4) Sorting by session/timestamp
      5) Basic NA handling for downstream model readiness
    """

    def __init__(self, cfg: Dict, *,
                 required_columns: Optional[List[str]] = None):
        self.cfg = cfg
        self.required_columns = required_columns or [
            cfg["data"]["timestamp_col"],
            cfg["data"]["session_col"],
            cfg["data"]["event_col"],
            "group_id",
            cfg["data"]["label_col"],
        ]
        self.event_fx = EventFeatureExtractor(event_col=cfg["data"]["event_col"])
        self.time_fx = TimeFeatureEngineer(
            timestamp_col=cfg["data"]["timestamp_col"],
            session_col=cfg["data"]["session_col"],
        )

    def _load_csv(self, file_path="data/code_challenge_dataset.csv") -> pd.DataFrame:
        """
        Loads a CSV from data directory.
        If the file has no header or mismatched columns, you can predefine names here.
        """
        df = pd.read_csv(
            file_path,
            encoding='latin1',
            on_bad_lines='skip',
            header=0,
            names=["timestamp", "session_id", "event_name", "group_id", "step_name"]
        )
        return df

    def _assert_columns(self, df: pd.DataFrame) -> None:
        """
        Check for missing required columns in the dataframe.
        """
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}. Present: {list(df.columns)}")

    def _basic_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies light NA handling for categorical & numeric features used downstream.
        """
        out = df.copy()
        for c in [self.cfg["data"]["event_col"], "group_id"]:
            if c in out.columns:
                out[c] = out[c].astype(str)
        return out

    def load_and_prepare(self, file_path) -> pd.DataFrame:
        """
        Full pipeline:
          - load
          - schema check
          - basic cleaning
          - feature extraction (strings)
          - time features
          - final sorting
        Returns a DataFrame ready for tokenization/encoding.
        """
        df = self._load_csv(file_path)
        self._assert_columns(df)
        df = self._basic_cleaning(df)

        # 1) event-name features
        df = self.event_fx.transform(df)

        # 2) time features
        df = self.time_fx.transform(df)

        # 3) NA handling for model convenience
        for c in ["action_type", "target_app", "file_extension"]:
            if c in df.columns:
                df[c] = df[c].fillna("unknown")
        for c in ["first_number", "file_path_depth"]:
            if c in df.columns:
                df[c] = df[c].fillna(-1)
        if "is_sharepoint" in df.columns:
            df["is_sharepoint"] = df["is_sharepoint"].fillna(False)

        # 4) final sort
        df = df.sort_values(
            by=[self.cfg["data"]["session_col"], self.cfg["data"]["timestamp_col"]]
        ).reset_index(drop=True)

        return df
