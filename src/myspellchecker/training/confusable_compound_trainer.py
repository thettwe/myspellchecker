"""Confusable/compound MLP classifier trainer.

Trains a binary classifier to detect confusable word pairs and broken
compounds. Follows the same pattern as reranker_trainer.py: load JSONL,
train a 2-layer MLP, export to ONNX with normalization stats.

Usage:
    python -m myspellchecker.training.confusable_compound_trainer \
        --data data/confusable_compound_training.jsonl \
        --output data/confusable_compound_classifier.onnx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

NUM_FEATURES = 22
FEATURE_NAMES = [
    "log_freq_w1",
    "log_freq_w2",
    "log_freq_compound",
    "freq_ratio",
    "compound_valid",
    "bigram_prob_pair",
    "pmi",
    "npmi",
    "pos_w1_is_verb",
    "pos_w1_is_noun",
    "pos_w2_is_particle",
    "pos_w2_is_noun",
    "is_title_suffix",
    "is_a_prefix",
    "is_reduplication",
    "syl_w1",
    "syl_w2",
    "char_len_compound",
    "bigram_prob_left",
    "bigram_prob_right",
    "freq_imbalance",
    "compound_dominant",
]


def load_data(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load JSONL training data into feature matrix and label vector."""
    features_list = []
    labels_list = []
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            features_list.append(record["features"])
            labels_list.append(float(record["label"]))
    return np.array(features_list, dtype=np.float32), np.array(labels_list, dtype=np.float32)


def compute_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and std for z-score normalization."""
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean, std


def train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    hidden_size: int = 32,
    lr: float = 0.001,
    epochs: int = 100,
    patience: int = 10,
):
    """Train a 2-layer MLP with early stopping."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("ERROR: PyTorch required. Install with: pip install torch")
        sys.exit(1)

    class BinaryMLP(nn.Module):
        def __init__(self, input_size, hidden):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)
            self.fc2 = nn.Linear(hidden, 1)

        def forward(self, x):
            return self.fc2(self.dropout(self.relu(self.fc1(x)))).squeeze(-1)

    model = BinaryMLP(x_train.shape[1], hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    x_t = torch.tensor(x_train)
    y_t = torch.tensor(y_train)
    x_v = torch.tensor(x_val)
    y_v = torch.tensor(y_val)

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(x_t), y_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x_v)
            val_loss = criterion(val_logits, y_v).item()
            preds = (torch.sigmoid(val_logits) > 0.5).float()
            tp = ((preds == 1) & (y_v == 1)).sum().item()
            fp = ((preds == 1) & (y_v == 0)).sum().item()
            fn = ((preds == 0) & (y_v == 1)).sum().item()
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(
                f"  Epoch {epoch:3d}: loss={loss.item():.4f} "
                f"val_loss={val_loss:.4f} F1={f1:.3f} P={prec:.3f} R={rec:.3f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model


def export_onnx(model, mean: np.ndarray, std: np.ndarray, output_path: str) -> None:
    """Export to ONNX with stats JSON."""
    import torch

    model.eval()
    onnx_path = Path(output_path)
    torch.onnx.export(
        model,
        torch.randn(1, NUM_FEATURES),
        str(onnx_path),
        input_names=["features"],
        output_names=["logit"],
        dynamic_axes={"features": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=13,
    )
    print(f"  ONNX: {onnx_path}")

    stats_path = onnx_path.with_suffix(".stats.json")
    with open(stats_path, "w") as f:
        json.dump(
            {
                "feature_names": FEATURE_NAMES,
                "feature_means": mean.tolist(),
                "feature_stds": std.tolist(),
                "num_features": NUM_FEATURES,
            },
            f,
            indent=2,
        )
    print(f"  Stats: {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="Train confusable/compound MLP")
    parser.add_argument("--data", required=True, help="Training JSONL")
    parser.add_argument("--output", required=True, help="Output ONNX path")
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--val-split", type=float, default=0.2)
    args = parser.parse_args()

    print(f"Loading {args.data}...")
    x, y = load_data(args.data)
    print(f"  {len(x)} samples ({int(y.sum())} pos, {int(len(y) - y.sum())} neg)")

    # Stratified split
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng = np.random.default_rng(42)
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    vp = int(len(pos_idx) * args.val_split)
    vn = int(len(neg_idx) * args.val_split)
    val_idx = np.concatenate([pos_idx[:vp], neg_idx[:vn]])
    train_idx = np.concatenate([pos_idx[vp:], neg_idx[vn:]])

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    print(
        f"  Train: {len(x_train)} ({int(y_train.sum())} pos)"
        f" | Val: {len(x_val)} ({int(y_val.sum())} pos)"
    )

    mean, std = compute_stats(x_train)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    print(f"\nTraining (hidden={args.hidden}, lr={args.lr})...")
    model = train_mlp(
        x_train,
        y_train,
        x_val,
        y_val,
        args.hidden,
        args.lr,
        args.epochs,
        args.patience,
    )

    # Final eval
    import torch

    model.eval()
    with torch.no_grad():
        preds = (torch.sigmoid(model(torch.tensor(x_val))) > 0.5).float().numpy()
    tp = int(((preds == 1) & (y_val == 1)).sum())
    fp = int(((preds == 1) & (y_val == 0)).sum())
    fn = int(((preds == 0) & (y_val == 1)).sum())
    tn = int(((preds == 0) & (y_val == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    print(f"\nFinal: TP={tp} FP={fp} FN={fn} TN={tn} P={prec:.3f} R={rec:.3f} F1={f1:.3f}")

    print(f"\nExporting to {args.output}...")
    export_onnx(model, mean, std, args.output)
    print("Done!")


if __name__ == "__main__":
    main()
