"""Configuration loader from YAML.

This module loads configuration settings from config.yaml and provides
access to all configuration parameters throughout the application.

Usage:
    from src.config.config import cfg
    data_csv = cfg["data"]["data_csv"]
"""
from pathlib import Path
import yaml

CONFIG_PATH = Path(__file__).parent / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)
