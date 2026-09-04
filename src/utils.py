"""Small shared utilities: reproducibility, output directory management, config loading."""

import os
import random
from datetime import datetime

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_run_dir(base_output_dir: str, run_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_output_dir, f"{run_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
