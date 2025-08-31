import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from typing import Dict, Any

import numpy as np

from cortical_column import CorticalColumn
from config import DEFAULT_CONFIG, CorticalConfig


def _stamp_metadata(config: CorticalConfig) -> Dict[str, Any]:
    from subprocess import check_output, CalledProcessError
    try:
        git_sha = check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        git_sha = "unknown"
    try:
        pip = sys.executable
        py_ver = check_output([pip, "-c", "import sys;print(sys.version)"]).decode().strip()
    except Exception:
        py_ver = sys.version
    return {
        "git_sha": git_sha,
        "python": py_ver,
        "config": config.to_dict(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
    }


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def exp_step_response(steps: int = 2000, size: int = 64, freq_hz: float = 0.0, amp: float = 1.0, seed: int = 42) -> Dict[str, Any]:
    np.random.seed(seed)
    column = CorticalColumn(size=size)
    dt = column.dt
    outputs = []
    l5_means = []
    # step at t=0: constant input
    for i in range(steps):
        t = i * dt
        if freq_hz > 0:
            sensory = amp * np.sin(2 * np.pi * freq_hz * t) * np.ones(size)
        else:
            sensory = amp * np.ones(size)
        column.step(sensory)
        outputs.append(np.mean(column.get_output()))
        l5_means.append(np.mean(column.layers['L5'].state))
    outputs = np.array(outputs)
    l5_means = np.array(l5_means)
    # crude tau estimate from 63% rise time on L5 mean
    final = l5_means[-1]
    target = 0.63 * final
    idx = int(np.argmax(l5_means >= target)) if np.any(l5_means >= target) else -1
    tau_est = idx * dt if idx >= 0 else None
    return {
        "dt": dt,
        "tau_est_L5_s": tau_est,
        "final_output_mean": float(outputs[-1]),
    }


def exp_frequency_sweep(size: int = 64, freqs=(1, 5, 10, 20, 50), steps: int = 2000, seed: int = 0) -> Dict[str, Any]:
    np.random.seed(seed)
    column = CorticalColumn(size=size)
    dt = column.dt
    results = {}
    for f in freqs:
        outs = []
        for i in range(steps):
            t = i * dt
            sensory = np.sin(2 * np.pi * f * t) * np.ones(size)
            column.step(sensory)
            if i > steps // 2:  # skip transient
                outs.append(np.mean(column.get_output()))
        results[str(f)] = float(np.std(outs)) if outs else 0.0
    return {"std_response": results}


def exp_noise_sweep(size: int = 64, noise_levels=(0.0, 0.1, 0.5, 1.0), steps: int = 500, seed: int = 0) -> Dict[str, Any]:
    np.random.seed(seed)
    column = CorticalColumn(size=size)
    dt = column.dt
    res = {}
    for nl in noise_levels:
        outs = []
        for i in range(steps):
            t = i * dt
            clean = np.sin(2 * np.pi * 10 * t) * np.ones(size)
            noisy = clean + np.random.normal(0, nl, size)
            column.step(noisy)
            outs.append(np.mean(column.get_output()))
        res[str(nl)] = {"mean": float(np.mean(outs)), "std": float(np.std(outs))}
    return {"noise_response": res}


EXPERIMENTS = {
    "step": exp_step_response,
    "freq": exp_frequency_sweep,
    "noise": exp_noise_sweep,
}


def exp_stability_long(size: int = 64, steps: int = 10000, seed: int = 0, amp: float = 1.0) -> Dict[str, Any]:
    """Long-run stability under sustained input.

    Tracks non-finite occurrences and state magnitudes across layers.
    """
    np.random.seed(seed)
    column = CorticalColumn(size=size)
    dt = column.dt
    clip = DEFAULT_CONFIG.integration.get('state_soft_clip', 10.0)

    per_layer_maxabs: Dict[str, float] = {name: 0.0 for name in column.layers.keys()}
    clipped_counts: Dict[str, int] = {name: 0 for name in column.layers.keys()}
    nonfinite_total = 0

    for i in range(steps):
        sensory = amp * np.ones(size)
        column.step(sensory)
        for name, layer in column.layers.items():
            st = layer.state
            # Non-finite tracking
            nonfinite = np.sum(~np.isfinite(st))
            nonfinite_total += int(nonfinite)
            # Max abs
            try:
                per_layer_maxabs[name] = float(max(per_layer_maxabs[name], float(np.max(np.abs(st)))))
            except Exception:
                pass
            # Clipped fraction (near soft clip bound)
            clipped_counts[name] += int(np.sum(np.abs(st) > 0.95 * clip))

    total_vals = steps * size
    clipped_fraction = {name: (clipped_counts[name] / float(total_vals)) for name in clipped_counts}
    metrics = {
        "dt": dt,
        "steps": steps,
        "any_nonfinite": nonfinite_total > 0,
        "nonfinite_total": nonfinite_total,
        "per_layer_maxabs": per_layer_maxabs,
        "clipped_fraction": clipped_fraction,
        "final_output_mean": float(np.mean(column.get_output()))
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Neuromorphic cortical column experiments")
    parser.add_argument("experiment", choices=list(EXPERIMENTS.keys()) + ["stability"]) 
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Run
    if args.experiment == "step":
        metrics = EXPERIMENTS["step"](steps=args.steps, size=args.size, seed=args.seed)
    elif args.experiment == "freq":
        metrics = EXPERIMENTS["freq"](size=args.size, steps=args.steps, seed=args.seed)
    elif args.experiment == "noise":
        metrics = EXPERIMENTS["noise"](size=args.size, steps=args.steps, seed=args.seed)
    elif args.experiment == "stability":
        metrics = exp_stability_long(size=args.size, steps=args.steps, seed=args.seed)
    else:
        raise ValueError("Unknown experiment")

    # Persist
    run_id = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    outdir = os.path.join(args.outdir, args.experiment, run_id)
    _ensure_dir(outdir)
    meta = _stamp_metadata(DEFAULT_CONFIG)
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump({"metrics": metrics, "meta": meta}, f, indent=2)
    print(f"Saved results to {outdir}")


if __name__ == "__main__":
    main()
