"""I/O utilities for loading and saving data."""

import json
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Union
import hashlib


def save_jsonl(data: List[Dict[str, Any]], filepath: Union[str, Path]) -> None:
    """Save list of dictionaries to JSONL file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def load_jsonl(filepath: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load JSONL file to list of dictionaries."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_parquet(df: pd.DataFrame, filepath: Union[str, Path]) -> None:
    """Save DataFrame to Parquet file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(filepath, index=False)


def load_parquet(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load DataFrame from Parquet file."""
    return pd.read_parquet(filepath)


def save_csv(df: pd.DataFrame, filepath: Union[str, Path]) -> None:
    """Save DataFrame to CSV file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)


def load_csv(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load DataFrame from CSV file."""
    return pd.read_csv(filepath)


def get_hashed_filename(config: Dict[str, Any], suffix: str = "") -> str:
    """Generate hashed filename from config parameters."""
    config_str = json.dumps(config, sort_keys=True)
    hash_obj = hashlib.md5(config_str.encode())
    hash_hex = hash_obj.hexdigest()[:8]
    return f"{hash_hex}{suffix}"