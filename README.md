# ML4SCI DeepLense — GSoC evaluation tests

Solutions to the [ML4SCI DeepLense](https://ml4sci.org/) GSoC evaluation tests: **Common Test I** (multi-class
classification of strong-lensing images) and **Specific Test VI** (super-resolution of strong-lensing images).

Every notebook is fully executed, so results, figures and notes can be read on GitHub without running anything.

## Results

### Common Test I — classify lenses as no substructure / subhalo (CDM) / vortex

Data: 30,000 training images (10,000 per class) and 7,500 provided validation images, 150×150, one channel.
The provided training set is split **90:10** (27,000 train / 3,000 validation, stratified, seed 42). The validation split is
used for every choice (epoch, test-time augmentation, final model). The provided 7,500-image set is kept as a held-out
**test set** and used once at the end.

| Model (held-out test, n = 7,500) | Macro ROC AUC (OvR) | Subhalo vs rest | Vortex vs rest | No sub. vs rest | Accuracy |
|---|---|---|---|---|---|
| ResNet-18, standard stem (notebook 02) | 0.9958 | 0.9928 | 0.9985 | 0.9962 | 96.80 % |
| **ResNet-18, high-resolution stem (notebook 03, final)** | **0.9967** | **0.9945** | **0.9991** | **0.9964** | **97.32 %** |

* The standard ResNet stem shrinks the 150×150 image to 38×38 before the first residual block. The final model uses a 3×3
  stride-1 first convolution, so the first block works at 75×75 and keeps the faint subhalo signal. Errors fall from 240 to 201,
  and subhalos mistaken for "no substructure" fall from 124 to 103.
* The gain is statistically significant on the test set: McNemar exact test p = 0.0005, and the paired-bootstrap 95 % CI of the
  AUC difference is [+0.0003, +0.0013] (`src/compare_significance.py`, `report/significance_03.json`).
* Training: ImageNet-initialised ResNet-18 (RGB filters summed into one channel), random flips and 90° rotations, AdamW,
  warm-up + cosine schedule, 30 epochs, 8-view test-time augmentation.

### Specific Test VI — super-resolution (in progress)

| Data | Split (train / valid / test) | Baseline on test | PSNR (dB) | SSIM |
|---|---|---|---|---|
| VI.A: 10,000 simulated pairs, 75×75 → 150×150 | 8,100 / 900 / 1,000 | bicubic | 41.62 | 0.9747 |
| VI.B: 300 real pairs (HSC → HST), 64×64 → 128×128 | 243 / 27 / 30 | bicubic + brightness fit | 29.11 | 0.6704 |

Notebook 04 checks the data and fixes the split. It also shows that VI.A low-resolution images match an anti-aliased bicubic
downscale plus signal-dependent noise, and that VI.B's ground-based images are about 3.6× brighter in the background than the
Hubble images. An EDSR-style network trained on VI.A and then fine-tuned on VI.B (notebook 05) is being trained, and this
section will be updated with its results.

## Notebooks

| Notebook | Content |
|---|---|
| [`01_common_test_data`](notebooks/01_common_test_data.ipynb) | Data checks, class statistics, duplicate / leakage check, shortcut check, fixed 90:10 split |
| [`02_baseline_resnet18`](notebooks/02_baseline_resnet18.ipynb) | ResNet-18 baseline, ROC curves, confusion matrix |
| [`03_hires_resnet18`](notebooks/03_hires_resnet18.ipynb) | High-resolution stem, model choice on validation, comparison on test |
| [`04_superres_data_baselines`](notebooks/04_superres_data_baselines.ipynb) | Test VI data checks, how LR images were made, split, baselines |

Shared code is in `src/` (`deeplense_utils.py` for Test I, `deeplense_sr.py` for Test VI). Figures and JSON results are in `report/`.

## Reproduce

```bash
bash src/download_data.sh          # Common Test I (1.0 GB), VI.A (509 MB) and VI.B (21 MB) from the public test document
# unzip each archive inside its own folder under data/raw/, then run the notebooks in order
```

Tested with Python 3, PyTorch 2.14 (CUDA), torchvision 0.29, scikit-learn 1.9, scikit-image 0.26, on one NVIDIA RTX 2000 Ada
(16 GB). Trained weights of the final Test I model: `weights/resnet18_hires_best.pt`
(load with `src/deeplense_utils.make_resnet18("hires")`).

## AI assistance

The code and notebooks were written with substantial help from an AI coding assistant (Claude, by Anthropic). I chose the tests
and the direction, ran all experiments, reviewed the code and results, and can explain every step.

## Author

Priyanshu Kumar — B.Tech CSE, IIITDM Kurnool — GitHub [@FasterThanAi](https://github.com/FasterThanAi)
