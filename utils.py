"""Utilities: config, device, checkpoint, plotting."""

import csv
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
import torch


def plot_training_comparison(
    base_checkpoint_dir: Union[str, Path],
    save_path: Optional[Union[str, Path]] = None,
):
    """
    Plot comparative training and validation loss from multiple projector types.
    Expects subdirs of base_checkpoint_dir (e.g. 'linear', 'qformer'), each
    containing loss_history.csv with columns: epoch, train_loss, val_loss.
    """
    import matplotlib.pyplot as plt

    base = Path(base_checkpoint_dir)
    if not base.is_dir():
        raise FileNotFoundError(f"Checkpoint base dir not found: {base}")

    # Collect (projector_type, epochs, train_losses, val_losses) from each subdir
    series = []
    for subdir in sorted(base.iterdir()):
        if not subdir.is_dir():
            continue
        csv_path = subdir / "loss_history.csv"
        if not csv_path.exists():
            continue
        epochs_plot, train_losses, val_losses = [], [], []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    epochs_plot.append(int(row["epoch"]))
                    train_losses.append(float(row["train_loss"]))
                    val_losses.append(float(row["val_loss"]))
                except (KeyError, ValueError):
                    continue
        if epochs_plot:
            series.append((subdir.name, epochs_plot, train_losses, val_losses))

    if not series:
        print("No loss_history.csv found in any subdir of", base)
        return

    plt.rcParams["font.family"] = "Times New Roman"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for label, epochs_plot, train_losses, val_losses in series:
        ax1.plot(epochs_plot, train_losses, marker="o", markersize=4, label=label)
        ax2.plot(epochs_plot, val_losses, marker="s", markersize=4, label=label)

    ax1.set_xlabel("Epoch", fontname="Times New Roman")
    ax1.set_ylabel("Loss", fontname="Times New Roman")
    ax1.set_title("Training loss (by projector type)", fontname="Times New Roman")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Epoch", fontname="Times New Roman")
    ax2.set_ylabel("Loss", fontname="Times New Roman")
    ax2.set_title("Validation loss (by projector type)", fontname="Times New Roman")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(Path(save_path), bbox_inches="tight")
    plt.show()


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML config."""
    with open(path) as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    path: str,
    **kwargs: Any,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {"epoch": epoch, "model_state_dict": model.state_dict(), **kwargs}
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(state, path)


def load_checkpoint(path: str, model: Optional[torch.nn.Module] = None, device: Optional[torch.device] = None):
    state = torch.load(path, map_location=device or "cpu")
    if model is not None and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=True)
    return state
