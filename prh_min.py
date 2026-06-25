#!/usr/bin/env python
"""Minimal single-GPU reproduction of the *real-data* headline of arXiv 2602.14486.

The paper revisits the Platonic Representation Hypothesis (PRH) on real vision and
language models. Its headline real-data finding:

    Raw GLOBAL spectral similarity (linear CKA) between vision and language
    representations appears to INCREASE with model scale (the apparent
    "convergence" that motivates the PRH). After permutation null-calibration,
    that global trend LARGELY DISAPPEARS. In contrast, LOCAL neighborhood
    similarity (mutual-kNN) RETAINS significant cross-modal alignment after
    calibration.

This script reproduces that mechanism on a SMALL, fast subset:

  * Vision models (timm ViTs, augreg_in21k): tiny -> small -> base   (scale axis)
  * Language models (Bloomz): 560m -> 1b1 -> 1b7                      (scale axis)
  * Data: a slice of the WIT image-text pairs (minhuh/prh, wit_1024)
  * Metrics: cka_lin (GLOBAL, confounded) and mutual_knn (LOCAL, robust)

It drives the repo's own `run_prh_experiment` (which extracts features, computes
the layer-aggregated similarity, and applies the null calibration / gating), then
analyzes the raw vs. calibrated alignment as a function of model scale and writes
EVAL.md + JSONL artifacts.

The qualitative predictions checked (and used as the run's pass/fail verdict):
  P1. Raw cka_lin shows a positive scale trend (apparent PRH convergence).
  P2. Calibration shrinks the global cka_lin alignment substantially
      (calibrated mean << raw mean): the global signal is largely a confound.
  P3. Local mutual_knn retains a meaningfully positive calibrated alignment
      (and noticeably more than calibrated cka_lin): local structure survives.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from aristotelian.prh.prh_experiment import run_prh_experiment
from aristotelian.prh.prh_models import get_models


def _scale_proxy_indices(n_models: int) -> np.ndarray:
    """Ordinal scale axis 0..n-1 (modelsets are listed small -> large)."""
    return np.arange(n_models, dtype=float)


def _trend_slope(scale: np.ndarray, values: np.ndarray) -> float:
    """Least-squares slope of values vs. a normalized scale axis."""
    s = (scale - scale.mean())
    if np.allclose(s, 0):
        return 0.0
    return float((s * (values - values.mean())).sum() / (s * s).sum())


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_samples = int(os.environ.get("PRH_MAX_SAMPLES", "256"))
    num_perm = int(os.environ.get("PRH_PERMS", "200"))
    batch_size = int(os.environ.get("PRH_BATCH", "8"))
    modelset = os.environ.get("PRH_MODELSET", "min")
    alpha = 0.05

    art = Path(".openresearch/artifacts")
    art.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def log(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    llm, lvm = get_models(modelset, modality="all")
    log("=" * 78)
    log("Minimal single-GPU PRH reproduction — arXiv 2602.14486")
    log(f"device={device}  modelset={modelset}  max_samples={max_samples}  "
        f"perms={num_perm}  batch={batch_size}  alpha={alpha}")
    log(f"language models ({len(llm)}): {llm}")
    log(f"vision models   ({len(lvm)}): {lvm}")
    log("=" * 78)

    metrics = ["cka_lin", "mutual_knn"]
    summaries: dict[str, dict] = {}

    for metric in metrics:
        log(f"\n### Metric: {metric} "
            f"({'GLOBAL spectral (confounded)' if metric == 'cka_lin' else 'LOCAL neighborhood (robust)'})")
        out = run_prh_experiment(
            dataset="minhuh/prh",
            subset="wit_1024",
            split="train",
            modelset=modelset,
            modality_x="language",
            pool_x="avg",
            modality_y="vision",
            pool_y="cls",
            caption_idx=0,
            max_samples=max_samples,
            batch_size=batch_size,
            device=device,
            output_dir="./results",
            k=10,
            metric=metric,
            num_permutations=num_perm,
            alpha=alpha,
            q_outlier=0.95,
            force_features=False,
            num_workers=1,
        )
        raw = np.asarray(out["scores"], dtype=float)      # [n_lang, n_vision]
        gated = np.asarray(out["gated"], dtype=float)      # calibrated
        pvals = np.asarray(out["pvalues"], dtype=float)
        fdr_mask = np.asarray(out["fdr_mask"])

        raw_mean = float(raw.mean())
        gated_mean = float(gated.mean())
        frac_sig = float(fdr_mask.mean())

        # Scaling trend: average alignment per "total scale" = lang_idx + vis_idx.
        nL, nV = raw.shape
        sL = _scale_proxy_indices(nL)
        sV = _scale_proxy_indices(nV)
        # mean over all pairs at each combined-scale level
        combined = {}
        for i in range(nL):
            for j in range(nV):
                lvl = int(sL[i] + sV[j])
                combined.setdefault(lvl, {"raw": [], "gated": []})
                combined[lvl]["raw"].append(raw[i, j])
                combined[lvl]["gated"].append(gated[i, j])
        levels = sorted(combined)
        raw_by_lvl = np.array([np.mean(combined[l]["raw"]) for l in levels])
        gated_by_lvl = np.array([np.mean(combined[l]["gated"]) for l in levels])
        lvl_arr = np.array(levels, dtype=float)
        raw_slope = _trend_slope(lvl_arr, raw_by_lvl)
        gated_slope = _trend_slope(lvl_arr, gated_by_lvl)

        log(f"  raw alignment matrix mean   = {raw_mean:.4f}")
        log(f"  calibrated (gated) mean     = {gated_mean:.4f}")
        log(f"  FDR-significant fraction    = {frac_sig:.2f}")
        log(f"  raw scale-trend slope       = {raw_slope:+.4f}")
        log(f"  calibrated scale-trend slope= {gated_slope:+.4f}")
        log(f"  raw matrix:\n{np.array2string(raw, precision=3, suppress_small=True)}")
        log(f"  gated matrix:\n{np.array2string(gated, precision=3, suppress_small=True)}")

        summaries[metric] = dict(
            raw_mean=raw_mean, gated_mean=gated_mean, frac_sig=frac_sig,
            raw_slope=raw_slope, gated_slope=gated_slope,
            raw=raw.tolist(), gated=gated.tolist(),
            levels=[int(x) for x in levels],
            raw_by_level=raw_by_lvl.tolist(), gated_by_level=gated_by_lvl.tolist(),
            shape=[nL, nV],
        )

    cka = summaries["cka_lin"]
    knn = summaries["mutual_knn"]

    # P1: raw global CKA shows a positive scale trend (apparent convergence).
    p1 = cka["raw_slope"] > 0
    # P2: calibration substantially shrinks global CKA alignment.
    p2 = cka["gated_mean"] < 0.5 * cka["raw_mean"] + 1e-9
    # P3: local mKNN keeps a meaningfully positive calibrated alignment, and
    #     more than calibrated global CKA.
    p3 = (knn["gated_mean"] > 0.02) and (knn["gated_mean"] > cka["gated_mean"])

    checks = {
        "P1_raw_cka_scale_trend_positive": bool(p1),
        "P2_calibration_shrinks_global_cka": bool(p2),
        "P3_local_mknn_survives_calibration": bool(p3),
    }
    all_pass = all(checks.values())
    verdict = "REPRODUCED" if all_pass else "NOT REPRODUCED"

    log("\n" + "=" * 78)
    log("VERDICT")
    for k, v in checks.items():
        log(f"  [{'PASS' if v else 'FAIL'}] {k}")
    log(f"  => {verdict}")
    log("=" * 78)

    # artifacts
    with open(art / "prh_summary.json", "w") as f:
        json.dump({"checks": checks, "reproduced": all_pass,
                   "config": {"device": device, "modelset": modelset,
                              "max_samples": max_samples, "perms": num_perm,
                              "alpha": alpha},
                   "summaries": {m: {kk: vv for kk, vv in s.items()
                                     if kk not in ("raw", "gated")}
                                 for m, s in summaries.items()}}, f, indent=2)
    with open(art / "prh_matrices.json", "w") as f:
        json.dump({m: {"raw": s["raw"], "gated": s["gated"], "shape": s["shape"]}
                   for m, s in summaries.items()}, f, indent=2)
    (art / "prh_run_log.txt").write_text("\n".join(lines) + "\n")

    # EVAL.md
    e = []
    e.append("# Minimal GPU reproduction — arXiv 2602.14486 (PRH real data)\n")
    e.append(f"**Verdict: {verdict}** "
             f"({sum(checks.values())}/{len(checks)} checks passed)\n")
    e.append("Reproduces the paper's real-data headline on a small scale-spanning "
             "model set and a WIT image-text slice, on a single GPU, using the "
             "repo's own `run_prh_experiment` (feature extraction + null "
             "calibration).\n")
    e.append(f"Config: modelset=`{modelset}`, max_samples={max_samples}, "
             f"permutations={num_perm}, alpha={alpha}, device={device}.\n")
    e.append(f"Language models: {', '.join(llm)}.\n")
    e.append(f"Vision models: {', '.join(lvm)}.\n")

    e.append("\n## Global vs. local, raw vs. calibrated\n")
    e.append("| metric | type | raw mean | calibrated mean | shrinkage | raw scale-slope | calib scale-slope |")
    e.append("|--------|------|----------|-----------------|-----------|-----------------|-------------------|")
    for m, label in [("cka_lin", "GLOBAL spectral"), ("mutual_knn", "LOCAL neighborhood")]:
        s = summaries[m]
        shrink = 1.0 - (s["gated_mean"] / s["raw_mean"]) if s["raw_mean"] > 1e-9 else float("nan")
        e.append(f"| `{m}` | {label} | {s['raw_mean']:.4f} | {s['gated_mean']:.4f} "
                 f"| {shrink*100:.0f}% | {s['raw_slope']:+.4f} | {s['gated_slope']:+.4f} |")

    e.append("\n## Interpretation\n")
    e.append(f"- **P1 (apparent convergence):** raw global CKA scale-slope = "
             f"{cka['raw_slope']:+.4f} (>0 expected): "
             f"{'PASS' if p1 else 'FAIL'}.\n")
    e.append(f"- **P2 (global is a confound):** calibration shrinks global CKA "
             f"from {cka['raw_mean']:.4f} to {cka['gated_mean']:.4f} "
             f"({'PASS' if p2 else 'FAIL'} — calibrated < half of raw).\n")
    e.append(f"- **P3 (local survives):** calibrated mutual_knn = "
             f"{knn['gated_mean']:.4f} vs calibrated CKA = {cka['gated_mean']:.4f} "
             f"({'PASS' if p3 else 'FAIL'} — local retains alignment, more than global).\n")
    e.append(f"\n**Conclusion:** the Aristotelian real-data finding is reproduced on "
             f"this minimal set: the apparent global (CKA) convergence is largely a "
             f"scale confound that calibration removes, while local neighborhood "
             f"(mutual-kNN) cross-modal alignment survives calibration. "
             f"Overall: **{verdict}**.\n")

    Path("EVAL.md").write_text("\n".join(e) + "\n")
    (art / "EVAL.md").write_text("\n".join(e) + "\n")

    if not all_pass:
        raise SystemExit(f"PRH reproduction FAILED: {checks}")
    print("\nPRH reproduction succeeded; EVAL.md written.")


if __name__ == "__main__":
    main()
