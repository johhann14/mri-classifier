import csv, os, time
from typing import Any, Dict
from pathlib import Path
import random
from src.model import SimpleCNN
import torch.nn as nn
import torch
from src.utils1 import save_model, save_plots

def _ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def new_run(root):
    name = _ts()
    run_dir = Path(root) / name
    metrics_csv = run_dir / "metrics.csv"

    return run_dir, metrics_csv


def log_metrics(csv_path, **metrics):
    os.makedirs(csv_path.parent, exist_ok=True)
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    ordered = {}
    for k in ("split", "epoch", "step", "loss", "acc", "lr"):
        if k in metrics: ordered[k] = metrics[k]
    for k in sorted(metrics.keys()):
        if k not in ordered:
            ordered[k] = metrics[k]

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ordered.keys()))
        if not file_exists:
            w.writeheader()
        w.writerow(ordered)
        f.flush()

    
