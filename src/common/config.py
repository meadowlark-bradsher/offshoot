"""Configuration management with OmegaConf."""

from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from typing import Optional


def load_config(config_path: str, overrides: Optional[list] = None) -> DictConfig:
    """Load configuration from YAML file with optional overrides."""
    config = OmegaConf.load(config_path)

    if overrides:
        override_config = OmegaConf.from_dotlist(overrides)
        config = OmegaConf.merge(config, override_config)

    return config


def save_config(config: DictConfig, output_path: str) -> None:
    """Save configuration to YAML file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output_path)


def get_output_dir(config: DictConfig, create: bool = True) -> Path:
    """Get output directory from config, creating if needed."""
    output_dir = Path(config.logging.out_dir) / config.experiment_name
    if create:
        output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir