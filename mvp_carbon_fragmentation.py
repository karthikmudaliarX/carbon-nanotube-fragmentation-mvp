#!/usr/bin/env python3
"""
MVP: Carbon Cluster Fragmentation Analysis
============================================
Based on: "Experimental and Theoretical Aspects of the Fragmentation of
Carbon's Single and Multi-Walled Nanotubes" (Javeed & Ahmad, 2025)

This script:
  1. Simulates (or loads) energy-stepped mass spectra of sputtered Cx clusters
     (x = 1..4) for pristine and damaged carbon nanotubes.
  2. Computes absolute and normalized sputtering yields.
  3. Infers local thermal-spike temperature Ts from cluster ratios via an
     Arrhenius-like model.
  4. Calculates Shannon entropy and KL divergence of the cluster probability
     distributions.
  5. Estimates a Higuchi fractal dimension of the probability series.
  6. Produces CSV outputs (out_mvp/) and matplotlib plots.

Usage:
    python mvp_carbon_fragmentation.py          # run full analysis with simulated data
    python mvp_carbon_fragmentation.py --test   # run embedded tests only
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
KB_EV = 8.617333262e-5  # Boltzmann constant in eV/K
EPS = 1e-30  # epsilon guard against log(0) / division-by-zero

# Default binding-energy differences (heuristic, eV) used for Ts estimation
DEFAULT_DELTA_E: Dict[Tuple[int, int], float] = {
    (2, 1): 0.4,
    (2, 3): 0.9,
    (3, 4): 0.6,
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Spectrum:
    """Single mass spectrum at a given ion energy."""
    energy_keV: float
    sample_type: str          # e.g. "pristine" or "damaged"
    material: str             # e.g. "SWCNT" or "MWCNT"
    counts: Dict[int, float]  # cluster size x -> raw counts


@dataclass
class Dataset:
    """Collection of spectra for one sample type."""
    sample_type: str
    material: str
    spectra: List[Spectrum] = field(default_factory=list)

    @property
    def energies(self) -> np.ndarray:
        return np.array([s.energy_keV for s in self.spectra])

    @property
    def cluster_sizes(self) -> List[int]:
        if not self.spectra:
            return []
        return sorted(self.spectra[0].counts.keys())


# ---------------------------------------------------------------------------
# Data simulation
# ---------------------------------------------------------------------------

def simulate_dataset(
    sample_type: str = "pristine",
    material: str = "SWCNT",
    energies: Optional[np.ndarray] = None,
    cluster_range: Tuple[int, int] = (1, 4),
    rng_seed: int = 42,
) -> Dataset:
    """Generate synthetic sputtering data for demonstration purposes.

    The model uses heuristic exponential decays so that:
      - C1 dominates at low energy, C2 peaks at intermediate energy,
        and larger clusters grow with energy.
      - "damaged" samples show broader distributions (higher entropy).
    """
    rng = np.random.default_rng(rng_seed)
    if energies is None:
        energies = np.linspace(0.2, 2.0, 10)

    x_min, x_max = cluster_range
    sizes = list(range(x_min, x_max + 1))

    # Heuristic parameters per cluster size
    # (amplitude, energy_peak, width)
    if sample_type == "pristine":
        params = {
            1: (1000, 0.4, 0.3),
            2: (600, 0.8, 0.4),
            3: (200, 1.2, 0.5),
            4: (60, 1.6, 0.6),
        }
    else:  # damaged
        params = {
            1: (800, 0.5, 0.5),
            2: (700, 0.9, 0.6),
            3: (400, 1.1, 0.7),
            4: (150, 1.4, 0.8),
        }

    spectra: List[Spectrum] = []
    for E in energies:
        counts: Dict[int, float] = {}
        for x in sizes:
            amp, peak, width = params.get(x, (100, 1.0, 0.5))
            mean_val = amp * np.exp(-0.5 * ((E - peak) / width) ** 2)
            # Add Poisson-like noise
            noisy = max(0.0, mean_val + rng.normal(0, max(1, np.sqrt(mean_val))))
            counts[x] = noisy
        spectra.append(Spectrum(energy_keV=E, sample_type=sample_type,
                                material=material, counts=counts))
    return Dataset(sample_type=sample_type, material=material, spectra=spectra)


def load_dataset_from_csv(
    path: str,
    cluster_range: Tuple[int, int] = (1, 4),
) -> Dataset:
    """Load a dataset from a CSV file.

    Expected columns: energy_keV, sample_type, material, C1, C2, C3, C4
    """
    df = pd.read_csv(path)
    required = {"energy_keV", "sample_type", "material"}
    x_min, x_max = cluster_range
    for x in range(x_min, x_max + 1):
        required.add(f"C{x}")
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    sample_type = str(df["sample_type"].iloc[0])
    material = str(df["material"].iloc[0])
    spectra: List[Spectrum] = []
    for _, row in df.iterrows():
        counts = {x: float(row[f"C{x}"]) for x in range(x_min, x_max + 1)}
        spectra.append(Spectrum(
            energy_keV=float(row["energy_keV"]),
            sample_type=str(row["sample_type"]),
            material=str(row["material"]),
            counts=counts,
        ))
    return Dataset(sample_type=sample_type, material=material, spectra=spectra)


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------

def compute_yields(ds: Dataset) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute absolute and normalized sputtering yields.

    Returns (yields_df, norm_df) where each row is an energy step and
    columns are cluster sizes.
    """
    sizes = ds.cluster_sizes
    energies = ds.energies
    rows_abs: List[Dict] = []
    rows_norm: List[Dict] = []

    for spec in ds.spectra:
        total = sum(spec.counts.values()) + EPS  # dose proxy
        row_a: Dict[str, float] = {"energy_keV": spec.energy_keV}
        row_n: Dict[str, float] = {"energy_keV": spec.energy_keV}
        abs_vals = {x: spec.counts[x] / total for x in sizes}
        sum_abs = sum(abs_vals.values()) + EPS
        for x in sizes:
            row_a[f"Y_C{x}"] = abs_vals[x]
            row_n[f"N_C{x}"] = abs_vals[x] / sum_abs
        rows_abs.append(row_a)
        rows_norm.append(row_n)

    return pd.DataFrame(rows_abs), pd.DataFrame(rows_norm)


def compute_probabilities(norm_df: pd.DataFrame, sizes: List[int]) -> pd.DataFrame:
    """Probability distribution p(Cx|E) — identical to normalized yields."""
    prob_df = norm_df.copy()
    prob_df.columns = [c.replace("N_C", "p_C") if c.startswith("N_C") else c
                       for c in prob_df.columns]
    return prob_df


def shannon_entropy(probs: np.ndarray) -> float:
    """H(p) = -sum p_i ln p_i  (natural log, base e)."""
    p = np.asarray(probs, dtype=float).ravel()
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return -float(np.sum(p * np.log(p + EPS)))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """D_KL(p || q) = sum p_i ln(p_i / q_i)."""
    p = np.asarray(p, dtype=float).ravel()
    q = np.asarray(q, dtype=float).ravel()
    # Guard: only where both > 0
    mask = (p > 0) & (q > 0)
    if not np.any(mask):
        return 0.0
    return float(np.sum(p[mask] * np.log((p[mask] + EPS) / (q[mask] + EPS))))


def estimate_thermal_spike_temperature(
    prob_df: pd.DataFrame,
    sizes: List[int],
    delta_e: Optional[Dict[Tuple[int, int], float]] = None,
) -> pd.DataFrame:
    """Estimate Ts from cluster-ratio Arrhenius model.

    Ts = -ΔE / (kB * ln(R))  where R = p(A) / p(B).
    """
    if delta_e is None:
        delta_e = DEFAULT_DELTA_E

    records: List[Dict] = []
    for _, row in prob_df.iterrows():
        E = row["energy_keV"]
        for (a, b), dE in delta_e.items():
            pa = row.get(f"p_C{a}", 0.0)
            pb = row.get(f"p_C{b}", 0.0)
            if pb < EPS or pa < EPS:
                Ts = np.nan
            else:
                R = pa / pb
                ln_R = np.log(R + EPS)
                if abs(ln_R) < 1e-12:
                    Ts = np.nan
                else:
                    Ts = -dE / (KB_EV * ln_R)
            records.append({
                "energy_keV": E,
                "ratio": f"C{a}/C{b}",
                "delta_E_eV": dE,
                "R": pa / (pb + EPS),
                "Ts_K": Ts,
            })
    return pd.DataFrame(records)


def higuchi_fractal_dimension(series: np.ndarray, k_max: int = 0) -> float:
    """Estimate fractal dimension via Higuchi's method.

    Parameters
    ----------
    series : 1-D array of length N
    k_max  : maximum interval; 0 = auto (N // 4, clamped to [2, 20])

    Returns
    -------
    D_f : float  (slope of log(L(k)) vs log(1/k))
    """
    N = len(series)
    if N < 4:
        return np.nan

    if k_max <= 0:
        k_max = max(2, min(N // 4, 20))

    ks = np.arange(1, k_max + 1)
    Lk = np.zeros(len(ks))

    for idx, k in enumerate(ks):
        lengths = []
        for m in range(1, k + 1):
            # indices: m-1, m-1+k, m-1+2k, ...
            idxs = np.arange(m - 1, N, k)
            if len(idxs) < 2:
                continue
            seg = series[idxs]
            n_seg = len(idxs)
            norm = (N - 1) / (k * (n_seg - 1) * k + EPS)
            length = norm * np.sum(np.abs(np.diff(seg)))
            lengths.append(length)
        Lk[idx] = np.mean(lengths) if lengths else 0.0

    # Linear fit in log-log space
    valid = Lk > 0
    if np.sum(valid) < 2:
        return np.nan

    x = np.log(1.0 / ks[valid])
    y = np.log(Lk[valid])
    coeffs = np.polyfit(x, y, 1)
    return float(coeffs[0])


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_normalized_yields(
    norm_pristine: pd.DataFrame,
    norm_damaged: pd.DataFrame,
    sizes: List[int],
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, (df, label) in zip(axes, [(norm_pristine, "Pristine"), (norm_damaged, "Damaged")]):
        for x in sizes:
            ax.plot(df["energy_keV"], df[f"N_C{x}"], "o-", label=f"C{x}")
        ax.set_xlabel("Ion Energy (keV)")
        ax.set_ylabel("Normalized Yield")
        ax.set_title(f"{label} — Normalized Yields")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_entropy_and_kl(
    prob_pristine: pd.DataFrame,
    prob_damaged: pd.DataFrame,
    sizes: List[int],
) -> plt.Figure:
    energies_p = prob_pristine["energy_keV"].values
    energies_d = prob_damaged["energy_keV"].values

    H_p = []
    H_d = []
    for _, row in prob_pristine.iterrows():
        probs = np.array([row[f"p_C{x}"] for x in sizes])
        H_p.append(shannon_entropy(probs))
    for _, row in prob_damaged.iterrows():
        probs = np.array([row[f"p_C{x}"] for x in sizes])
        H_d.append(shannon_entropy(probs))

    # KL divergence at each energy (assuming same energy grid)
    n_common = min(len(energies_p), len(energies_d))
    kl_vals = []
    for i in range(n_common):
        p = np.array([prob_pristine.iloc[i][f"p_C{x}"] for x in sizes])
        q = np.array([prob_damaged.iloc[i][f"p_C{x}"] for x in sizes])
        kl_vals.append(kl_divergence(p, q))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(energies_p, H_p, "s-", label="Pristine", color="tab:blue")
    ax1.plot(energies_d, H_d, "^-", label="Damaged", color="tab:red")
    ax1.set_xlabel("Ion Energy (keV)")
    ax1.set_ylabel("Shannon Entropy H (nats)")
    ax1.set_title("Shannon Entropy of Cluster Distribution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(energies_p[:n_common], kl_vals, "D-", color="tab:green")
    ax2.set_xlabel("Ion Energy (keV)")
    ax2.set_ylabel("D_KL(pristine || damaged)")
    ax2.set_title("KL Divergence: Pristine vs Damaged")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_thermal_spike(temp_pristine: pd.DataFrame, temp_damaged: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, (df, label) in zip(axes, [(temp_pristine, "Pristine"), (temp_damaged, "Damaged")]):
        for ratio_label in df["ratio"].unique():
            sub = df[df["ratio"] == ratio_label]
            ax.plot(sub["energy_keV"], sub["Ts_K"], "o-", label=ratio_label)
        ax.set_xlabel("Ion Energy (keV)")
        ax.set_ylabel("Ts (K)")
        ax.set_title(f"{label} — Thermal Spike Temperature")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------

def run_analysis(show_plots: bool = True) -> None:
    """Run the full MVP analysis pipeline."""
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_mvp")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Simulate datasets
    ds_pristine = simulate_dataset("pristine", "SWCNT", rng_seed=42)
    ds_damaged = simulate_dataset("damaged", "MWCNT", rng_seed=99)
    sizes = ds_pristine.cluster_sizes

    # 2. Compute yields
    yields_p, norm_p = compute_yields(ds_pristine)
    yields_d, norm_d = compute_yields(ds_damaged)

    yields_p.to_csv(os.path.join(out_dir, "pristine_yields.csv"), index=False)
    yields_d.to_csv(os.path.join(out_dir, "damaged_yields.csv"), index=False)

    # 3. Probabilities
    prob_p = compute_probabilities(norm_p, sizes)
    prob_d = compute_probabilities(norm_d, sizes)

    prob_p.to_csv(os.path.join(out_dir, "pristine_probs.csv"), index=False)
    prob_d.to_csv(os.path.join(out_dir, "damaged_probs.csv"), index=False)

    # 4. Thermal spike temperature
    temp_p = estimate_thermal_spike_temperature(prob_p, sizes)
    temp_d = estimate_thermal_spike_temperature(prob_d, sizes)

    temp_p.to_csv(os.path.join(out_dir, "pristine_temp.csv"), index=False)
    temp_d.to_csv(os.path.join(out_dir, "damaged_temp.csv"), index=False)

    # 5. Fractal dimension of C2 probability series
    c2_series_p = prob_p["p_C2"].values
    c2_series_d = prob_d["p_C2"].values
    Df_p = higuchi_fractal_dimension(c2_series_p)
    Df_d = higuchi_fractal_dimension(c2_series_d)

    df_frac_p = pd.DataFrame([{"cluster": "C2", "Df": Df_p}])
    df_frac_d = pd.DataFrame([{"cluster": "C2", "Df": Df_d}])
    df_frac_p.to_csv(os.path.join(out_dir, "pristine_Df_C2.csv"), index=False)
    df_frac_d.to_csv(os.path.join(out_dir, "damaged_Df_C2.csv"), index=False)

    # 6. Console summary
    print("=" * 60)
    print("Carbon Cluster Fragmentation Analysis — MVP Results")
    print("=" * 60)

    valid_Ts_p = temp_p["Ts_K"].dropna()
    valid_Ts_d = temp_d["Ts_K"].dropna()
    positive_Ts_p = valid_Ts_p[valid_Ts_p > 0]
    positive_Ts_d = valid_Ts_d[valid_Ts_d > 0]

    if len(positive_Ts_p) > 0:
        print(f"Pristine mean Ts (positive only): {positive_Ts_p.mean():.0f} K "
              f"(n={len(positive_Ts_p)} estimates)")
    else:
        print("Pristine: no valid positive Ts estimates")

    if len(positive_Ts_d) > 0:
        print(f"Damaged  mean Ts (positive only): {positive_Ts_d.mean():.0f} K "
              f"(n={len(positive_Ts_d)} estimates)")
    else:
        print("Damaged: no valid positive Ts estimates")

    print(f"Pristine Higuchi D_f (C2): {Df_p:.4f}")
    print(f"Damaged  Higuchi D_f (C2): {Df_d:.4f}")
    print(f"\nCSV outputs saved to: {out_dir}")
    print("=" * 60)

    # 7. Plots
    if show_plots:
        fig1 = plot_normalized_yields(norm_p, norm_d, sizes)
        fig2 = plot_entropy_and_kl(prob_p, prob_d, sizes)
        fig3 = plot_thermal_spike(temp_p, temp_d)

        # Save plots as PNG as well
        fig1.savefig(os.path.join(out_dir, "normalized_yields.png"), dpi=150, bbox_inches="tight")
        fig2.savefig(os.path.join(out_dir, "entropy_kl.png"), dpi=150, bbox_inches="tight")
        fig3.savefig(os.path.join(out_dir, "thermal_spike.png"), dpi=150, bbox_inches="tight")
        print("Plots saved to out_mvp/*.png")

        plt.show()


# ---------------------------------------------------------------------------
# Embedded tests
# ---------------------------------------------------------------------------

def run_tests() -> None:
    """Run minimal embedded tests."""
    print("Running tests...")
    passed = 0
    failed = 0

    # --- test_compute_yields ---
    try:
        ds = simulate_dataset("pristine", "SWCNT", rng_seed=0)
        yields_df, norm_df = compute_yields(ds)
        assert "energy_keV" in yields_df.columns, "Missing energy_keV in yields"
        assert "Y_C1" in yields_df.columns, "Missing Y_C1 in yields"
        # Check normalization: sum of N_Cx ≈ 1 for each row
        for _, row in norm_df.iterrows():
            total = sum(row[f"N_C{x}"] for x in [1, 2, 3, 4])
            assert abs(total - 1.0) < 1e-6, f"Normalization failed: {total}"
        print("  [PASS] test_compute_yields")
        passed += 1
    except AssertionError as e:
        print(f"  [FAIL] test_compute_yields: {e}")
        failed += 1
    except Exception as e:
        print(f"  [FAIL] test_compute_yields: {e}")
        failed += 1

    # --- test_shannon_entropy ---
    try:
        # Delta distribution: all mass on one state => H = 0
        H = shannon_entropy(np.array([1.0, 0.0, 0.0, 0.0]))
        assert abs(H) < 1e-10, f"Expected H≈0 for delta, got {H}"
        # Uniform distribution: H = ln(4)
        H_uniform = shannon_entropy(np.array([0.25, 0.25, 0.25, 0.25]))
        assert abs(H_uniform - np.log(4)) < 1e-6, f"Expected H≈ln(4), got {H_uniform}"
        print("  [PASS] test_shannon_entropy")
        passed += 1
    except AssertionError as e:
        print(f"  [FAIL] test_shannon_entropy: {e}")
        failed += 1
    except Exception as e:
        print(f"  [FAIL] test_shannon_entropy: {e}")
        failed += 1

    # --- test_kl_divergence ---
    try:
        p = np.array([0.25, 0.25, 0.25, 0.25])
        kl = kl_divergence(p, p)
        assert abs(kl) < 1e-10, f"Expected KL(p||p)≈0, got {kl}"
        # KL(p||q) >= 0 (Gibbs' inequality)
        q = np.array([0.1, 0.2, 0.3, 0.4])
        kl2 = kl_divergence(p, q)
        assert kl2 >= -1e-10, f"KL should be non-negative, got {kl2}"
        print("  [PASS] test_kl_divergence")
        passed += 1
    except AssertionError as e:
        print(f"  [FAIL] test_kl_divergence: {e}")
        failed += 1
    except Exception as e:
        print(f"  [FAIL] test_kl_divergence: {e}")
        failed += 1

    # --- test_higuchi_fractal_dimension ---
    try:
        # Constant series => D_f should be finite (close to 1 or nan-safe)
        series = np.ones(20)
        Df = higuchi_fractal_dimension(series)
        # For constant series, all diffs are 0 => Lk=0 => nan is acceptable
        # Just check it doesn't crash
        assert isinstance(Df, float), f"Expected float, got {type(Df)}"

        # Random series should give D_f roughly between 1 and 2
        rng = np.random.default_rng(123)
        series2 = rng.random(50)
        Df2 = higuchi_fractal_dimension(series2)
        assert 0.5 < Df2 < 2.5, f"D_f out of expected range: {Df2}"
        print("  [PASS] test_higuchi_fractal_dimension")
        passed += 1
    except AssertionError as e:
        print(f"  [FAIL] test_higuchi_fractal_dimension: {e}")
        failed += 1
    except Exception as e:
        print(f"  [FAIL] test_higuchi_fractal_dimension: {e}")
        failed += 1

    # --- test_thermal_spike_temperature ---
    try:
        ds = simulate_dataset("pristine", "SWCNT", rng_seed=7)
        _, norm_df = compute_yields(ds)
        prob_df = compute_probabilities(norm_df, ds.cluster_sizes)
        temp_df = estimate_thermal_spike_temperature(prob_df, ds.cluster_sizes)
        assert "Ts_K" in temp_df.columns, "Missing Ts_K column"
        assert len(temp_df) > 0, "Empty temperature dataframe"
        print("  [PASS] test_thermal_spike_temperature")
        passed += 1
    except AssertionError as e:
        print(f"  [FAIL] test_thermal_spike_temperature: {e}")
        failed += 1
    except Exception as e:
        print(f"  [FAIL] test_thermal_spike_temperature: {e}")
        failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    if failed > 0:
        print("SOME TESTS FAILED")
        sys.exit(1)
    else:
        print("All tests passed")
        sys.exit(0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Carbon Cluster Fragmentation Analysis MVP"
    )
    parser.add_argument("--test", action="store_true", help="Run embedded tests and exit")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot display")
    args = parser.parse_args()

    if args.test:
        run_tests()
    else:
        run_analysis(show_plots=not args.no_plots)


if __name__ == "__main__":
    main()
