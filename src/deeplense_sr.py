"""Shared code for DeepLense Specific Test VI (super-resolution): data, split, metrics, model, training, prediction."""
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity

ROOT = Path("~/gsoc/eval-tests/ml4sci-deeplense").expanduser()
PROC = ROOT / "data/processed/superres"


# ---------------------------------------------------------------- data
def load_pairs(which):
    """which = "a" (simulated, 75 -> 150) or "b" (real HSC -> HST, 64 -> 128).
    Returns LR (N,h,w) float32, HR (N,2h,2w) float32 and the LR file names, sorted by sample number."""
    base = ROOT / f"data/raw/superres_{which}/Dataset"
    lr_files = [f for f in os.listdir(base / "LR") if f.endswith(".npy")]
    hr_files = [f for f in os.listdir(base / "HR") if f.endswith(".npy")]
    if which == "a":
        num = lambda f: int(f[len("sample"):-4])
        to_hr = lambda f: f
    else:
        num = lambda f: int(f[len("LR_"):-4])
        to_hr = lambda f: "HR_" + f[len("LR_"):]
    lr_files = sorted(lr_files, key=num)
    assert sorted(map(to_hr, lr_files)) == sorted(hr_files), "LR and HR files do not pair up one-to-one"
    LR = np.stack([np.load(base / "LR" / f)[0] for f in lr_files]).astype(np.float32)
    HR = np.stack([np.load(base / "HR" / to_hr(f))[0] for f in lr_files]).astype(np.float32)
    assert HR.shape[1] == 2 * LR.shape[1] and HR.shape[2] == 2 * LR.shape[2]
    assert np.isfinite(LR).all() and np.isfinite(HR).all()
    return LR, HR, lr_files


def make_split(n, seed=42):
    """90:10 train/test split; 10 % of the train part is held out as a validation set for model choice."""
    perm = np.random.default_rng(seed).permutation(n)
    n_test = n // 10
    test, rest = perm[:n_test], perm[n_test:]
    n_valid = len(rest) // 10
    return dict(train=np.sort(rest[n_valid:]), valid=np.sort(rest[:n_valid]), test=np.sort(test))


def load_split(which):
    s = json.loads((PROC / f"split_{which}.json").read_text())
    return {k: np.array(v) for k, v in s.items() if k in ("train", "valid", "test")}


# ---------------------------------------------------------------- metrics
def upsample(x):
    """Bicubic 2x upsampling of a (N,1,h,w) tensor (the baseline, and the skip path of the network)."""
    return F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)


def upsample_np(LR):
    return upsample(torch.from_numpy(LR)[:, None]).numpy()[:, 0]


def metrics(SR, HR):
    """Per-image MSE, PSNR and SSIM. Intensities are on a 0-1 scale, so the data range is fixed at 1.0.
    Outputs are not clipped (real HR images go slightly above 1)."""
    mse = ((SR.astype(np.float64) - HR) ** 2).mean(axis=(1, 2))
    psnr = 10 * np.log10(1.0 / mse)
    ssim = np.array([structural_similarity(h, s, data_range=1.0) for s, h in zip(SR, HR)])
    return pd.DataFrame(dict(mse=mse, psnr=psnr, ssim=ssim))


def summarise(df):
    return pd.Series({"MSE": df.mse.mean(), "PSNR (dB)": df.psnr.mean(), "SSIM": df.ssim.mean()})


def bootstrap_diff(a, b, n=2000, seed=0):
    """Paired bootstrap of mean(a - b) over images: (mean, 2.5 %, 97.5 %, share of resamples > 0)."""
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    means = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return d.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5), (means > 0).mean()


# ---------------------------------------------------------------- augmentation
def dihedral(x, k):
    """k = 0..7: rotate by 90*(k % 4) degrees, mirrored first if k >= 4."""
    if k >= 4:
        x = x.flip(-1)
    return torch.rot90(x, k % 4, dims=(-2, -1))


def inv_dihedral(x, k):
    x = torch.rot90(x, -(k % 4), dims=(-2, -1))
    return x.flip(-1) if k >= 4 else x


def augment_pair(lr, hr):
    """The same random dihedral transform for each LR image and its HR partner."""
    ks = torch.randint(0, 8, (lr.shape[0],), device=lr.device)
    lo, hi = torch.empty_like(lr), torch.empty_like(hr)
    for k in range(8):
        m = ks == k
        if m.any():
            lo[m], hi[m] = dihedral(lr[m], k), dihedral(hr[m], k)
    return lo, hi


# ---------------------------------------------------------------- model
class ResBlock(nn.Module):
    def __init__(self, c, res_scale=0.1):
        super().__init__()
        self.conv1, self.conv2 = nn.Conv2d(c, c, 3, padding=1), nn.Conv2d(c, c, 3, padding=1)
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.res_scale * self.conv2(F.relu(self.conv1(x)))


class EDSR(nn.Module):
    """EDSR-style network (Lim et al. 2017): residual blocks without batch norm at low resolution, then a
    sub-pixel (PixelShuffle) 2x upsampler. The output is added to the bicubic upsampling of the input, and the last
    conv starts at zero, so the untrained network returns exactly the bicubic baseline and learns only the correction."""

    def __init__(self, n_blocks=16, channels=64):
        super().__init__()
        c = channels
        self.head = nn.Conv2d(1, c, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(c) for _ in range(n_blocks)], nn.Conv2d(c, c, 3, padding=1))
        self.up = nn.Sequential(nn.Conv2d(c, 4 * c, 3, padding=1), nn.PixelShuffle(2))
        self.out = nn.Conv2d(c, 1, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        f = self.head(x)
        f = f + self.body(f)
        return upsample(x) + self.out(self.up(f))


# ---------------------------------------------------------------- training and prediction
@torch.no_grad()
def predict_sr(model, LR, idx, device, tta=False, batch_size=64):
    """Super-resolved images for rows idx of LR; tta=True averages the 8 dihedral views (mapped back)."""
    model.eval()
    out = []
    for s in range(0, len(idx), batch_size):
        x = torch.from_numpy(LR[idx[s:s + batch_size]])[:, None].to(device)
        views = range(8) if tta else [0]
        y = sum(inv_dihedral(model(dihedral(x, k)), k) for k in views) / len(views)
        out.append(y[:, 0].float().cpu())
    return torch.cat(out).numpy()


def train_sr(model, LR, HR, idx_train, idx_valid, cfg, device, log=print):
    """L1 training with Adam, linear warm-up then cosine decay. Every cfg['eval_every'] epochs the validation PSNR
    is computed; the weights with the best validation PSNR are kept. Returns (history DataFrame, best dict)."""
    gen = torch.Generator().manual_seed(cfg["seed"])
    steps_per_epoch = int(np.ceil(len(idx_train) / cfg["batch_size"]))
    total, warm = cfg["epochs"] * steps_per_epoch, cfg["warmup_epochs"] * steps_per_epoch

    def lr_factor(s):
        if s < warm:
            return (s + 1) / warm
        return 0.5 * (1 + np.cos(np.pi * min(1.0, (s - warm) / max(1, total - warm))))

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    history, best = [], {"psnr": -np.inf}
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        t0, losses = time.time(), []
        order = idx_train[torch.randperm(len(idx_train), generator=gen).numpy()]
        for s in range(0, len(order), cfg["batch_size"]):
            b = order[s:s + cfg["batch_size"]]
            lr_b = torch.from_numpy(LR[b])[:, None].to(device, non_blocking=True)
            hr_b = torch.from_numpy(HR[b])[:, None].to(device, non_blocking=True)
            lr_b, hr_b = augment_pair(lr_b, hr_b)
            loss = F.l1_loss(model(lr_b), hr_b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        if epoch % cfg["eval_every"] == 0 or epoch == cfg["epochs"]:
            m = metrics(predict_sr(model, LR, idx_valid, device), HR[idx_valid])
            row = dict(epoch=epoch, train_l1=float(np.mean(losses)), valid_mse=m.mse.mean(), valid_psnr=m.psnr.mean(),
                       valid_ssim=m.ssim.mean(), lr=opt.param_groups[0]["lr"], sec=time.time() - t0)
            history.append(row)
            if row["valid_psnr"] > best["psnr"]:
                best = {"psnr": row["valid_psnr"], "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
            log(f"epoch {epoch:3d} | train L1 {row['train_l1']:.5f} | valid PSNR {row['valid_psnr']:.3f} dB "
                f"SSIM {row['valid_ssim']:.4f} | {row['sec']:.0f}s")
    return pd.DataFrame(history), best
