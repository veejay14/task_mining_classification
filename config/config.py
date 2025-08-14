"""
Config loader from YAML.
Usage:
    from config import cfg
    data_csv = cfg["data"]["data_csv"]
"""
from pathlib import Path
import yaml

CONFIG_PATH = Path(__file__).parent / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)
