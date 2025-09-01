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


def exp_l23_spectral(size: int = 32, steps: int = 5000, seed: int = 0) -> Dict[str, Any]:
    """Track L2/3 spectral radius during repeated-pattern learning.

    Every checkpoint, estimate spectral radius via power iteration.
    """
    np.random.seed(seed)
    column = CorticalColumn(size=size)
    layer = column.layers['L2/3']
    dt = column.dt

    # Simple repeated binary pattern
    pattern = (np.arange(size) % 2).astype(float)

    def estimate_radius(W: np.ndarray) -> float:
        v = np.random.randn(W.shape[0])
        v /= (np.linalg.norm(v) + 1e-9)
        for _ in range(5):
            v = W @ v
            n = np.linalg.norm(v) + 1e-9
            v = v / n
        return float(np.linalg.norm(W @ v))

    checkpoints = list(range(0, steps + 1, max(steps // 20, 1)))
    radius_series = {}
    for i in range(steps + 1):
        column.step(pattern)
        if i in checkpoints:
            r = estimate_radius(layer.hebbian_weights)
            radius_series[str(i)] = r

    mean_abs_delta_W = float(np.mean(np.abs(layer.hebbian_weights)))
    return {
        "dt": dt,
        "radius_series": radius_series,
        "mean_abs_weight": mean_abs_delta_W,
    }


def exp_pwm_duty_sweep(size: int = 32, blocks: int = 11, block_steps: int = 400, seed: int = 0) -> Dict[str, Any]:
    """Approximate PWM duty vs input amplitude monotonicity.

    For a set of constant amplitudes in [0, 1], run block_steps and
    approximate duty as the fraction of time the L5 motor output exceeds a threshold.
    """
    np.random.seed(seed)
    column = CorticalColumn(size=size)
    dt = column.dt
    amps = np.linspace(0.0, 1.0, blocks)
    duty_estimates = []
    threshold = 0.2  # heuristic threshold for 'high' output
    warmup = block_steps // 2
    for a in amps:
        highs = 0
        total = 0
        for i in range(block_steps):
            sensory = a * np.ones(size)
            column.step(sensory)
            if i >= warmup:
                mo = column.get_output()
                highs += int(np.mean(mo) > threshold)
                total += 1
        duty_estimates.append(highs / float(max(total, 1)))

    # Compute Spearman-like rank correlation without SciPy
    def ranks(x: np.ndarray) -> np.ndarray:
        return np.argsort(np.argsort(x))
    rho = float(np.corrcoef(ranks(amps), ranks(np.array(duty_estimates)))[0, 1])

    mapping = {str(round(a, 3)): float(d) for a, d in zip(amps, duty_estimates)}
    return {"dt": dt, "duty_mapping": mapping, "spearman_rho": rho}


def exp_l4_selectivity_map(size: int = 32, freqs=(2, 5, 10, 20, 40, 80), steps: int = 1500, seed: int = 0) -> Dict[str, Any]:
    """Map frequency → best-responding L4 neuron index and compare to expected.

    Uses per-neuron log-spaced centers from config range.
    """
    np.random.seed(seed)
    column = CorticalColumn(size=size)
    dt = column.dt
    f_low, f_high = DEFAULT_CONFIG.layers['L4'].frequency_range
    centers = np.logspace(np.log10(f_low), np.log10(f_high), size)

    results = {}
    for f in freqs:
        # run with sinusoidal input
        outputs = []
        for i in range(steps):
            t = i * dt
            sensory = np.sin(2 * np.pi * f * t) * np.ones(size)
            column.step(sensory)
            if i > steps // 2:  # collect during steady-state
                outputs.append(column.layers['L4'].output.copy())
        outs = np.array(outputs)
        # choose neuron index with max std
        stds = np.std(outs, axis=0)
        best_idx = int(np.argmax(stds))
        # expected index is nearest center frequency
        exp_idx = int(np.argmin(np.abs(centers - f)))
        results[str(f)] = {
            "best_idx": best_idx,
            "expected_idx": exp_idx,
            "index_error": abs(best_idx - exp_idx),
        }

    return {"l4_selectivity": results}


EXPERIMENTS = {
    "step": exp_step_response,
    "freq": exp_frequency_sweep,
    "noise": exp_noise_sweep,
    "l23_spectral": exp_l23_spectral,
    "pwm": exp_pwm_duty_sweep,
    "l4map": exp_l4_selectivity_map,
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
    elif args.experiment == "l23_spectral":
        metrics = EXPERIMENTS["l23_spectral"](size=args.size, steps=args.steps, seed=args.seed)
    elif args.experiment == "pwm":
        # blocks not exposed via CLI parser; use default inside function
        metrics = EXPERIMENTS["pwm"](size=args.size, seed=args.seed)
    elif args.experiment == "l4map":
        metrics = EXPERIMENTS["l4map"](size=args.size, steps=args.steps, seed=args.seed)
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
