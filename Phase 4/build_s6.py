"""build_s6.py -- Phase 4 S6: mean-logit ensemble of the five submitted runs (S1, S2, S5 from notebook 1;
S3, S4 from notebook 2). CPU only, runs in seconds.

Usage:
    python build_s6.py ens_nb1.json ens_nb2.json [--lib DIR] [--out DIR]

    ens_nb1.json / ens_nb2.json : written by p4_01_cjpe_notebook1 / p4_02_cjpe_notebook2 (dev + test logits of every run)
    --lib DIR                   : folder holding cj_lib.py (written by either notebook; default: current folder)
    --out DIR                   : output folder (default: ./p4_s6); writes S6/submission.zip, probs.json, metrics.json
"""
import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("ens_nb1")
ap.add_argument("ens_nb2")
ap.add_argument("--lib", default=".")
ap.add_argument("--out", default="p4_s6")
args = ap.parse_args()
sys.path.insert(0, str(Path(args.lib).resolve()))
import cj_lib as CJ  # noqa: E402

d1 = json.load(open(args.ens_nb1))
d2 = json.load(open(args.ens_nb2))
MEMBERS = {"S1": d1["S1"], "S2": d1["S2"], "S5": d1["S5"], "S3": d2["S3"], "S4": d2["S4"]}
dev_labels = d1["dev_labels"]
assert dev_labels == d2["dev_labels"], "the two notebooks must use the same dev set"
dev_ids = sorted(dev_labels)
test_ids = sorted(next(iter(MEMBERS.values()))["test"])
for name, m in MEMBERS.items():
    assert set(m["dev"]) == set(dev_ids), f"{name}: dev id mismatch"
    assert set(m["test"]) == set(test_ids), f"{name}: test id mismatch"

y = np.array([dev_labels[i] for i in dev_ids], int)
dev_logit = {n: np.array([m["dev"][i] for i in dev_ids]) for n, m in MEMBERS.items()}
test_logit = {n: np.array([m["test"][i] for i in test_ids]) for n, m in MEMBERS.items()}

per_member = {n: {"dev_macro_f1": CJ.macro_f1(y, dev_logit[n] > 0), "dev_auc": round(CJ.roc_auc(y, dev_logit[n]), 4)}
              for n in MEMBERS}

ens_dev = np.mean(np.stack([dev_logit[n] for n in MEMBERS]), axis=0)
ens_test = np.mean(np.stack([test_logit[n] for n in MEMBERS]), axis=0)

threshold, calib = CJ.calibrate(ens_dev, y)        # dev-fitted threshold, kept only if a 2-fold dev check agrees
report = CJ.binary_report(y, (ens_dev > threshold).astype(int))
auc = CJ.roc_auc(y, ens_dev)
pred_test = (ens_test > threshold).astype(int)

out = Path(args.out) / "S6"
out.mkdir(parents=True, exist_ok=True)
rows = [{"id": tid, "pred": int(p)} for tid, p in zip(test_ids, pred_test)]
assert len(rows) == len(test_ids) == len(set(test_ids))
json.dump(rows, open(out / "prediction.json", "w"))
with zipfile.ZipFile(out / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
    z.write(out / "prediction.json", arcname="prediction.json")
json.dump({"dev": {i: float(v) for i, v in zip(dev_ids, ens_dev)},
           "test": {i: float(v) for i, v in zip(test_ids, ens_test)},
           "threshold": threshold}, open(out / "probs.json", "w"))
metrics = {"name": "S6", "title": "Ensemble: mean logit of S1+S2+S3+S4+S5", "members": sorted(MEMBERS),
           "dev_report": report, "dev_auc": round(auc, 4), "threshold": threshold, "calibration": calib,
           "per_member_dev": per_member, "test_predicted_1_pct": round(100 * float(pred_test.mean()), 1)}
json.dump(metrics, open(out / "metrics.json", "w"), indent=1)

print("per-member dev scores:")
for n in ("S1", "S2", "S3", "S4", "S5"):
    print(f"  {n}: macro-F1 {per_member[n]['dev_macro_f1']:.4f} | AUC {per_member[n]['dev_auc']:.4f}")
print(f"\nS6 ensemble (mean logit, 5 members): dev macro-F1 {report['macro_f1']:.4f} "
      f"(P {report['macro_precision']:.4f} R {report['macro_recall']:.4f}) | dev AUC {auc:.4f}")
print(f"threshold used: {threshold:.4f} ({'2-fold dev-fitted' if calib['threshold_fitted'] else 'default 0'})")
print(f"test predicted 1: {metrics['test_predicted_1_pct']}%")
print(f"-> {out / 'submission.zip'}")
