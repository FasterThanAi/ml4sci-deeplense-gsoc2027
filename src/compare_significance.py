"""Is the hi-res model really better than the baseline on the test set? McNemar test + paired bootstrap of macro AUC."""
import json, sys
from pathlib import Path
import numpy as np, torch
from scipy.stats import binomtest
ROOT = Path("~/gsoc/eval-tests/ml4sci-deeplense").expanduser()
sys.path.insert(0, str(ROOT / "src"))
import deeplense_utils as U

dev = torch.device("cuda"); d = U.load_common()
mean, std = d["meta"]["pixel_mean_trainfull"], d["meta"]["pixel_std_trainfull"]
y = d["y_test"]; P = {}
for name, stem in [("baseline", "baseline"), ("hires", "hires")]:
    m = U.make_resnet18(stem).to(dev).to(memory_format=torch.channels_last)
    m.load_state_dict(torch.load(ROOT / f"weights/resnet18_{name}_best.pt", map_location=dev, weights_only=False)["state_dict"])
    P[name] = U.predict(m, d["X_test"], d["idx_test"], mean, std, dev, tta=True)
ok_b, ok_h = P["baseline"].argmax(1) == y, P["hires"].argmax(1) == y
b_only, h_only = int((ok_b & ~ok_h).sum()), int((~ok_b & ok_h).sum())   # discordant pairs
p_mcnemar = binomtest(h_only, b_only + h_only, 0.5).pvalue
rng = np.random.default_rng(0); diffs = []
for _ in range(2000):
    i = rng.integers(0, len(y), len(y))
    diffs.append(U.macro_auc(y[i], P["hires"][i]) - U.macro_auc(y[i], P["baseline"][i]))
diffs = np.array(diffs)
res = {"test_macro_auc": {k: U.macro_auc(y, p) for k, p in P.items()},
       "errors": {"baseline": int((~ok_b).sum()), "hires": int((~ok_h).sum())},
       "mcnemar": {"only_baseline_right": b_only, "only_hires_right": h_only, "exact_p_value": p_mcnemar},
       "bootstrap_auc_diff": {"mean": float(diffs.mean()), "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                              "share_of_resamples_hires_better": float((diffs > 0).mean()), "n_resamples": 2000}}
(ROOT / "report/significance_03.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
