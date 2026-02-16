# MVP Implementation Plan: Carbon Cluster Fragmentation Analysis

## Overview
This MVP ingests or simulates energy-stepped mass spectra of sputtered Cx clusters (x = 1..4), computes absolute and normalized sputtering yields, infers a local thermal spike temperature Ts from cluster ratios using an Arrhenius-like model, calculates Shannon entropy and KL divergence, and estimates a Higuchi fractal dimension. It produces CSV outputs and plots.

Strictness rules:
- All steps are explicitly actionable with expected inputs and outputs.
- No invented libraries or functions are referenced.
- The plan includes a requirements.txt and minimal tests embedded in the script.
- Code paths are guarded against edge cases (e.g., division by zero, empty data).

## Deliverables
- Executable script: mvp_carbon_fragmentation.py
- Data outputs (auto-created directory out_mvp/):
  - {pristine|damaged}_yields.csv
  - {pristine|damaged}_probs.csv
  - {pristine|damaged}_temp.csv
  - {pristine|damaged}_Df_C2.csv
- Plots (non-blocking windows): normalized yields, probabilities, entropy, Ts estimates, and KL divergence
- Minimal pass/fail tests (assertions) run automatically if the --test flag is used

## Dependencies
- Python 3.8+
- numpy
- pandas
- matplotlib

Install with:
```
pip install numpy pandas matplotlib
```

requirements.txt:
```
numpy>=1.21
pandas>=1.3
matplotlib>=3.5
```

## File Layout
- mvp_carbon_fragmentation.py
- requirements.txt
- out_mvp/ (auto-created; contains CSV outputs)

## Execution Steps
1) Run the MVP (simulation included; no external data required):
```
python mvp_carbon_fragmentation.py
```
Expected: produces plots and out_mvp/*.csv files. Console prints stats for mean Ts and fractal dimension.

2) Run tests (self-contained, no extra packages):
```
python mvp_carbon_fragmentation.py --test
```
Expected: prints “All tests passed” and exits with code 0.

3) Use your own CSV data:
- Prepare CSV with columns: energy_keV, sample_type, material, C1, C2, C3, C4
- Replace simulate_dataset calls with:
```
ds = load_dataset_from_csv("path/to/your/data.csv", cluster_range=(1,4))
```
- Keep cluster_range=(1,4) or adjust if your data include larger clusters; update plots and algorithms accordingly.

4) Outputs:
- CSV files are saved to ./out_mvp/ (auto-created).
- Plots open as windows; to switch to inline backend in Jupyter, set:
```
import matplotlib
matplotlib.use("module://matplotlib.backends.backend_agg")
```
before calling plotting functions.

## Algorithms Implemented
- Absolute and normalized yields:
  - Yx(E) = counts(Cx, E) / dose_like(E)
  - Nx(E) = Yx(E) / sum_x Yx(E)
- Probability distributions: p(Cx|E) = Nx(E)
- Shannon entropy: H(p) = -sum p_i ln p_i (base e)
- Relative entropy (KL divergence): D_KL(p||q) = sum p_i ln(p_i/q_i)
- Thermal spike temperature:
  - R = p(A)/p(B)
  - Ts = -ΔE / (kB ln R), with kB = 8.617333262e-5 eV/K and ΔE in eV
  - Default ΔE values (heuristic): { (C2,C1): 0.4 eV, (C2,C3): 0.9 eV, (C3,C4): 0.6 eV }
- Fractal dimension (Higuchi): D_f from slope of log(L(k)) vs log(1/k) for p(Cj) vs energy

## Configuration Parameters (hardened defaults)
- cluster_range = (1, 4) — change if your data extend beyond C4
- ΔE values for Ts estimation — adjust to calibrate with known thermochemistry or experiments
- energies = np.linspace(0.2, 2.0, 10) — override by passing energies to simulate_dataset or via CSV

## Minimal Tests Embedded in Script
- Test functions:
  - test_compute_yields: checks keys and normalization for synthetic data
  - test_shannon_entropy: verifies entropy = 0 for a delta distribution
  - test_kl_divergence: verifies KL(p||p) = 0
  - test_higuchi_fractal_dimension: ensures numeric stability for trivial series
- Run tests via: python mvp_carbon_fragmentation.py --test

## Known Caveats and Hardening
- Ts inference is heuristic and should be calibrated with known ΔE values for accurate temperatures.
- Dose proxy: the code uses total counts as a proxy for ion dose. If you have explicit dose D(E), divide counts by D(E) before computing yields.
- Division-by-zero protection is implemented (epsilon guards) to prevent NaNs/Inf.
- The fractal dimension via Higuchi is applied to p(Cj) vs energy as a complexity metric; it is not a direct measure of morphological fractal dimension.

## Troubleshooting
- Import errors: confirm dependencies with pip list | grep -E "(numpy|pandas|matplotlib)"
- No plots shown (headless server): ensure a display is available or switch to a non-interactive backend (see Outputs above).
- CSV loading errors: verify exact column names including case (energy_keV, sample_type, material, C1, C2, C3, C4).

## Extending to Real Data
- Add loader for SNICS CSV/mzXML if needed; keep the existing Spectrum/Dataset classes and functions unchanged.
- Calibrate ΔE using independent thermochemical data or MD-informed energetics for clusters.
- Expand cluster_range to include higher Cx clusters; adjust plots and Ts ratio pairs accordingly.