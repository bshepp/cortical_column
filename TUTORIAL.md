# Neuromorphic Cortical Column – From First Principles to Mastery

This tutorial is a hands-on path from the core ideas (analog integrators, time constants, field coupling) to running and interpreting reproducible experiments with this codebase. You’ll build intuition, verify behavior with metrics, and learn how to tune parameters confidently.

The guide assumes you’ve followed the README to set up the virtualenv and can run `make test` successfully.

---

## 0) First principles you’ll use

- Analog integrator dynamics (discrete Euler):
  - Continuous: τ dx/dt = −x + f(inputs)
  - Discrete: x[t+1] = x[t] + dt · (−x[t] + f(inputs[t])) / τ_s
  - In this code: dt is in seconds; all layer τ values are in milliseconds and converted internally to seconds.
- Field coupling vs. synaptic coupling:
  - Lateral matrices approximate distance-decaying interactions.
  - Field coupling injects cross-layer influence computed by `FieldCoupling`.
- Noise and stability:
  - White noise scales ∝ sqrt(dt).
  - Soft state clipping (tanh) bounds trajectories; recurrent spectral radius and gains regulate stability.
- Nonlinearities and carriers:
  - Sigmoid activation for layer outputs; PWM carrier in L5 driven by a time-based phase.

---

## 1) Orient yourself – architecture at a glance

- L4 (sensory front-end): per-neuron Butterworth bandpass filters + temporal edge detection.
- L2/3 (associative mesh): recurrent integrators + Hebbian learning; stability controls on recurrence.
- L5 (motor output): burst detection + leaky integration + PWM carrier.
- L6 (feedback/timing): oscillator advanced with dt; generates timing and resonance signals.
- L1 (modulation): contextual modulation and delay line; applied top-down to L2/3 and L5.

All pathway gains and most thresholds live in `config.py` under `DEFAULT_CONFIG.integration` and `DEFAULT_CONFIG.layers`.

---

## 2) Reproducible runs and provenance

All experiment runs write to `results/<experiment>/<timestamp>/metrics.json` and include:
- Git SHA of the code you ran
- Python version
- A full configuration snapshot

This lets you replicate any result later.

---

## 3) Your first experiment: Step response (L5 τ and amplitude)

Goal: sanity-check time constants and pathway gains.

Run:
```bash
make exp-step
```
Open `results/step/<timestamp>/metrics.json` and note:
- `tau_est_L5_s`: rough 63% rise-time estimate for L5 state mean
- `final_output_mean`: steady-state motor output mean

Exercises:
1) Increase `integration.l5_integration_rate` by +0.2 and re-run. Observe how `tau_est_L5_s` changes.
2) Lower `layers['L5'].burst_threshold` (e.g., 0.05 → 0.02). Does steady-state output increase (more frequent bursts)?
3) Decrease `integration.l5_from_l23_gain`. How does coupling from L2/3 affect step response?

Expected: Faster integration rate usually reduces effective rise time; lower burst threshold increases duty/bursts (up to saturation).

---

## 4) Frequency sweep: Is L4 selective?

Goal: verify frequency selectivity and propagation to output.

Run:
```bash
make exp-freq
```
Open `results/freq/<timestamp>/metrics.json`:
- `std_response`: a dictionary of frequency → output variability (higher ≈ more response)

Exercises:
1) Narrow L4 bandwidth by changing its center-band creation (optional) or reduce coupling into L5 with `integration.l5_from_l4_gain`.
2) Change `layers['L4'].frequency_range` from default (1–100 Hz) to (2–60 Hz) and re-run. Which frequencies now dominate?
3) Keep frequency range but set `integration.l23_from_l4_gain` lower. Does the response flatten?

Expected: Peaks around band centers; off-band frequencies show lower variability.

---

## 5) Noise robustness: Behavior across noise levels

Goal: quantify stability under input noise.

Run:
```bash
make exp-noise
```
Open `results/noise/<timestamp>/metrics.json`:
- `noise_response`: per noise level → `{ mean, std }`

Exercises:
1) Increase `layers['L2/3'].noise_level` moderately. Does `std` increase smoothly?
2) Toggle `integration.noise_on_state` to false (adds noise post-activation). Compare results – which looks more realistic?
3) Increase `integration.activation_sharpness` (less steep sigmoid). Does it desensitize noise, reducing variance?

Expected: Variance grows with noise; no NaNs or numerical blow-ups.

---

## 6) Long-run stability (10k steps): Will it hold?

Goal: ensure no divergence or recurrent runaway.

Run:
```bash
make exp-stability
```
Open `results/stability/<timestamp>/metrics.json`:
- `any_nonfinite`: must be false
- `per_layer_maxabs`: largest |state| per layer (should be below `integration.state_soft_clip`)
- `clipped_fraction`: fraction of states near the soft-clip bound (ideally low)

Exercises:
1) Increase `integration.l23_recurrent_gain` from 0.8 to 0.95. Does `clipped_fraction` rise? If `any_nonfinite` ever becomes true, reduce gains or lower `l23_weight_radius`.
2) Reduce `integration.state_soft_clip` to 6.0. Do outputs compress more? Tradeoffs between saturation and stability become visible.
3) Increase `layers['L2/3'].learning_rate`. Watch stability and whether `per_layer_maxabs` rises.

Expected: Stability remains; heavier recurrence or learning increases clipping and demands stronger spectral radius control.

---

## 7) (Manual) Learning behavior – observe weight adaptation

Goal: see L2/3 Hebbian learning adapting to a pattern.

Run an interactive snippet in the project venv (e.g., `python`):
```python
import numpy as np
from cortical_column import CorticalColumn

col = CorticalColumn(size=16)
initial = col.layers['L2/3'].hebbian_weights.copy()
pattern = np.array([1,0,1,0,1,0,1,0] * 2)

for _ in range(1000):
    col.step(pattern)

final = col.layers['L2/3'].hebbian_weights
print('mean |ΔW|:', np.mean(np.abs(final - initial)))
```
Exercises:
1) Increase `layers['L2/3'].weight_decay` to 0.2. Does `mean |ΔW|` decrease (stronger decay)?
2) Add noise to the pattern during training; observe robustness.
3) Lower `integration.l23_weight_radius` (e.g., 0.8). Does it stabilize learning but dampen adaptation?

Expected: Non-zero weight change that grows with learning rate and drops with weight decay.

---

## 8) (Manual) L1 modulation – top-down influence

Goal: measure L1’s effect on L2/3 and L5.

Snippet:
```python
import numpy as np
from cortical_column import CorticalColumn

col = CorticalColumn(size=16)
base = []
mod = []
for _ in range(200):
    col.step(np.ones(16))
    base.append(np.mean(col.layers['L5'].state))

# Apply stronger L1 modulation gain
from config import DEFAULT_CONFIG
DEFAULT_CONFIG.integration['l1_modulation_gain'] = 0.4
for _ in range(200):
    col.step(np.ones(16))
    mod.append(np.mean(col.layers['L5'].state))

print('ΔL5 mean with stronger L1:', np.mean(mod) - np.mean(base))
```
Exercises:
1) Lower `l1_modulation_gain` to 0.1; verify reduced influence on L5.
2) Increase `layers['L1'].noise_level`; observe whether modulation becomes noisier.

---

## 9) dt and τ: Avoiding integrator instability

- Guidance: keep `simulation.dt` ≤ 0.1 × min(τ_ms)/1000.
- If you bump `dt`, monitor `any_nonfinite` in stability runs and consider lowering path gains.

Exercises:
1) Double `simulation.dt` to 0.002. Does stability degrade? Restore to 0.001 after testing.
2) Increase smallest τ (e.g., L4 10 → 15 ms) and assess stability and responsiveness.

---

## 10) Advanced exercises

- Implement a custom experiment: response to multi-tone inputs; log band-wise output.
- Plot figures directly from stored metrics; compare runs across commits.
- Extend L4 to allow configurable Q-factor per neuron; validate via frequency sweeps.
- Add an experiment that reports the estimated spectral radius of L2/3 weights over time.

---

## 11) Reproducibility & sharing

- Each run is tied to a Git SHA and config snapshot. When you change parameters, commit them and re-run to track evolution.
- For publications: bundle `results/` with the code commit and `requirements.txt` to enable one-click reproduction.

Have fun experimenting—and treat the metrics as your compass. When results surprise you, adjust one variable at a time and re-run the corresponding experiment to isolate effects.
