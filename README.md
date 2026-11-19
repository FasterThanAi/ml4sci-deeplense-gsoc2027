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

### Specific Test VI — super-resolution (2× upscaling)

Data: VI.A has 10,000 simulated pairs, 75×75 → 150×150, split 8,100 / 900 / 1,000 (train / valid / test). VI.B has 300 real
pairs (ground-based HSC → Hubble), 64×64 → 128×128, split 243 / 27 / 30. Every choice (epoch, test-time augmentation, final
VI.B variant) is made on the validation set; the test sets are used once.

| Held-out test | Method | MSE | PSNR (dB) | SSIM |
|---|---|---|---|---|
| VI.A (n = 1,000) | bicubic | 6.94e-5 | 41.62 | 0.9747 |
| VI.A | **EDSR + 8-view TTA (final)** | **5.81e-5** | **42.39** | **0.9778** |
| VI.B (n = 30) | bicubic | 1.79e-2 | 20.25 | 0.3058 |
| VI.B | bicubic + brightness fit | 2.47e-3 | 29.11 | 0.6704 |
| VI.B | EDSR from VI.A, no fine-tuning | 2.41e-3 | 29.24 | 0.6915 |
| VI.B | EDSR trained from scratch on VI.B | 2.00e-3 | 32.35 | 0.7975 |
| VI.B | **EDSR fine-tuned from VI.A (final)** | 2.13e-3 | **32.41** | **0.7998** |

* **Model:** EDSR-style network (16 residual blocks, 64 channels, 1.37 M parameters, sub-pixel upsampler). It predicts a
  correction to bicubic upsampling, and starts out exactly equal to bicubic. L1 loss, Adam, warm-up + cosine schedule,
  random flips and 90° rotations applied to each pair.
* **VI.A:** +0.77 dB over bicubic (paired bootstrap 95 % CI [+0.76, +0.78]); better on all 1,000 test images. Notebook 04
  found signal-dependent noise in the LR images, which limits how much any model can recover.
* **VI.B:** the VI.A model alone hardly transfers (+0.14 dB over the brightness-fit baseline), because real HSC images
  differ from simulated ones in blur, noise and background. Fine-tuning on the 243 real pairs gives **+3.30 dB** over that
  baseline (95 % CI [+2.17, +4.42]).
* **VI.A pre-training made VI.B training faster but not better:** after 10 epochs the fine-tuned model is at 33.0 dB on
  validation vs 31.4 dB from scratch, but the final test difference is +0.06 dB (95 % CI [−0.06, +0.17]), within noise.
  More realistic simulated degradation is the most promising way to make transfer from simulation pay off.

Notebook 04 checks the data, finds how the VI.A LR images were made, fixes the split and scores baselines. Notebook 05
trains and evaluates the models. All Test VI training ran on the CPU because the shared GPU was busy.

## Notebooks

| Notebook | Content |
|---|---|
| [`01_common_test_data`](notebooks/01_common_test_data.ipynb) | Data checks, class statistics, duplicate / leakage check, shortcut check, fixed 90:10 split |
| [`02_baseline_resnet18`](notebooks/02_baseline_resnet18.ipynb) | ResNet-18 baseline, ROC curves, confusion matrix |
| [`03_hires_resnet18`](notebooks/03_hires_resnet18.ipynb) | High-resolution stem, model choice on validation, comparison on test |
| [`04_superres_data_baselines`](notebooks/04_superres_data_baselines.ipynb) | Test VI data checks, how LR images were made, split, baselines |
| [`05_superres_edsr`](notebooks/05_superres_edsr.ipynb) | EDSR on VI.A, transfer to VI.B (no fine-tuning / fine-tuned / from scratch), significance tests |

Shared code is in `src/` (`deeplense_utils.py` for Test I, `deeplense_sr.py` for Test VI). Figures and JSON results are in `report/`.

## Reproduce

```bash
bash src/download_data.sh          # Common Test I (1.0 GB), VI.A (509 MB) and VI.B (21 MB) from the public test document
# unzip each archive inside its own folder under data/raw/, then run the notebooks in order
```

Tested with Python 3, PyTorch 2.14 (CUDA), torchvision 0.29, scikit-learn 1.9, scikit-image 0.26, on one NVIDIA RTX 2000 Ada
(16 GB). Trained weights of the final Test I model: `weights/resnet18_hires_best.pt`
(load with `src/deeplense_utils.make_resnet18("hires")`). Test VI weights: `weights/edsr_superres_a_best.pt` and
`weights/edsr_superres_b_finetuned.pt` (load with `src/deeplense_sr.EDSR()`).

## AI assistance

The code and notebooks were written with substantial help from an AI coding assistant (Claude, by Anthropic). I chose the tests
and the direction, ran all experiments, reviewed the code and results, and can explain every step.

## Author

Priyanshu Kumar — B.Tech CSE, IIITDM Kurnool — GitHub [@FasterThanAi](https://github.com/FasterThanAi)
