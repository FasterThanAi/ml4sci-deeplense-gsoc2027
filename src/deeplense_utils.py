"""Shared code for the DeepLense Common Test I notebooks (data batches, augmentation, models, training, prediction)."""
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.metrics import roc_auc_score

ROOT = Path("~/gsoc/eval-tests/ml4sci-deeplense").expanduser()
PROC = ROOT / "data/processed/common"


def load_common():
    """Arrays and the fixed split saved by notebook 01."""
    meta = json.loads((PROC / "split.json").read_text())
    d = dict(
        meta=meta,
        X_all=np.load(PROC / "trainfull_X.npy"), y_all=np.load(PROC / "trainfull_y.npy"),
        X_test=np.load(PROC / "test_X.npy"), y_test=np.load(PROC / "test_y.npy"),
        idx_train=np.array(meta["train_idx"]), idx_valid=np.array(meta["valid_idx"]),
    )
    d["idx_test"] = np.arange(len(d["y_test"]))
    assert len(np.intersect1d(d["idx_train"], d["idx_valid"])) == 0
    return d


def batches(X, y, idx, batch_size, mean, std, device, shuffle=False, gen=None):
    """Yield standardised (x, y) GPU batches for rows `idx` of X (y may be None)."""
    order = idx[torch.randperm(len(idx), generator=gen).numpy()] if shuffle else idx
    for s in range(0, len(order), batch_size):
        b = order[s:s + batch_size]
        xb = torch.from_numpy(X[b]).to(device, non_blocking=True)
        yb = torch.from_numpy(y[b]).to(device) if y is not None else None
        yield (xb - mean) / std, yb


def dihedral(x, k):
    """k = 0..7: rotate by 90*(k % 4) degrees, mirrored first if k >= 4."""
    if k >= 4:
        x = x.flip(-1)
    return torch.rot90(x, k % 4, dims=(-2, -1))


def augment(x):
    """Apply one random dihedral transform per image."""
    ks = torch.randint(0, 8, (x.shape[0],), device=x.device)
    out = torch.empty_like(x)
    for k in range(8):
        m = ks == k
        if m.any():
            out[m] = dihedral(x[m], k)
    return out


def make_resnet18(stem="baseline", n_classes=3):
    """ImageNet-initialised ResNet-18 for 1-channel 150x150 images.

    stem="baseline": original 7x7 stride-2 conv + max-pool (first block works at 38x38).
    stem="hires":    3x3 stride-1 conv + max-pool (first block works at 75x75, 2x more spatial detail).
    In both cases the RGB weights of the pretrained first conv are summed into one input channel; for "hires" the
    centre 3x3 of each pretrained 7x7 kernel is used as the starting point.
    """
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    grey = m.conv1.weight.data.sum(dim=1, keepdim=True)            # (64, 1, 7, 7)
    if stem == "baseline":
        m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        m.conv1.weight.data.copy_(grey)
    elif stem == "hires":
        m.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.conv1.weight.data.copy_(grey[:, :, 2:5, 2:5])
    else:
        raise ValueError(stem)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m


@torch.no_grad()
def predict(model, X, idx, mean, std, device, tta=False, batch_size=256):
    """Softmax probabilities for rows idx of X; with tta=True, averaged over the 8 dihedral views."""
    model.eval()
    out = []
    for xb, _ in batches(X, None, idx, batch_size, mean, std, device):
        xb = xb.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            views = range(8) if tta else [0]
            p = sum(F.softmax(model(dihedral(xb, k)).float(), dim=1) for k in views) / len(views)
        out.append(p.cpu())
    return torch.cat(out).numpy()


def macro_auc(y, p):
    return roc_auc_score(y, p, multi_class="ovr", average="macro")


def train(model, d, cfg, device, log=print):
    """Train on d['idx_train'], pick the epoch with the best validation macro AUC. Returns (history, best)."""
    mean, std = d["meta"]["pixel_mean_trainfull"], d["meta"]["pixel_std_trainfull"]
    X, y, idx_train, idx_valid = d["X_all"], d["y_all"], d["idx_train"], d["idx_valid"]
    gen = torch.Generator().manual_seed(cfg["seed"])
    steps_per_epoch = int(np.ceil(len(idx_train) / cfg["batch_size"]))
    total, warm = cfg["epochs"] * steps_per_epoch, cfg["warmup_epochs"] * steps_per_epoch

    def lr_factor(s):  # linear warm-up, then cosine decay to 0 at the last step
        if s < warm:
            return (s + 1) / warm
        return 0.5 * (1 + np.cos(np.pi * min(1.0, (s - warm) / max(1, total - warm))))

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    history, best = [], {"auc": -1.0}
    y_val = y[idx_valid]
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        t0, losses = time.time(), []
        for xb, yb in batches(X, y, idx_train, cfg["batch_size"], mean, std, device, shuffle=True, gen=gen):
            xb = augment(xb).contiguous(memory_format=torch.channels_last)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        p_val = predict(model, X, idx_valid, mean, std, device)
        row = dict(epoch=epoch, train_loss=float(np.mean(losses)),
                   valid_loss=F.nll_loss(torch.log(torch.from_numpy(p_val).clamp_min(1e-8)), torch.from_numpy(y_val)).item(),
                   valid_acc=float((p_val.argmax(1) == y_val).mean()), valid_auc=macro_auc(y_val, p_val),
                   lr=opt.param_groups[0]["lr"], sec=time.time() - t0)
        history.append(row)
        if row["valid_auc"] > best["auc"]:
            best = {"auc": row["valid_auc"], "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
        log(f"epoch {epoch:2d} | train loss {row['train_loss']:.4f} | valid loss {row['valid_loss']:.4f} "
            f"acc {row['valid_acc']:.4f} AUC {row['valid_auc']:.4f} | {row['sec']:.0f}s")
    return pd.DataFrame(history), best
