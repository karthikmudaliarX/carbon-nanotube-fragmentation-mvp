# Carbon Cluster Fragmentation Analysis — MVP

**Based on:** [Experimental and Theoretical Aspects of the Fragmentation of Carbon's Single and Multi-Walled Nanotubes](https://arxiv.org/pdf/2511.15467v1)  
**Authors:** Sumera Javeed, Shoaib Ahmad (2025)

## Overview

This MVP implements a computational analysis pipeline for energy-stepped mass spectra of sputtered carbon clusters (C₁–C₄) from pristine and damaged carbon nanotubes. It demonstrates key analytical techniques from the paper:

1. **Sputtering Yield Computation** — absolute and normalized yields Yₓ(E) and Nₓ(E)
2. **Probability Distributions** — p(Cₓ|E) for cluster formation at each ion energy
3. **Thermal Spike Temperature** — Arrhenius-like inference of local temperature Ts from cluster ratios
4. **Shannon Entropy & KL Divergence** — information-theoretic characterization of cluster distributions
5. **Higuchi Fractal Dimension** — complexity metric for the probability series

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full analysis (with simulated data)
python mvp_carbon_fragmentation.py

# Run without plot windows (saves PNGs only)
python mvp_carbon_fragmentation.py --no-plots

# Run embedded tests
python mvp_carbon_fragmentation.py --test
```

## Outputs

All outputs are saved to `out_mvp/`:

| File | Description |
|------|-------------|
| `pristine_yields.csv` | Absolute sputtering yields for pristine SWCNT |
| `damaged_yields.csv` | Absolute sputtering yields for damaged MWCNT |
| `pristine_probs.csv` | Cluster probability distributions (pristine) |
| `damaged_probs.csv` | Cluster probability distributions (damaged) |
| `pristine_temp.csv` | Thermal spike temperature estimates (pristine) |
| `damaged_temp.csv` | Thermal spike temperature estimates (damaged) |
| `pristine_Df_C2.csv` | Higuchi fractal dimension for C₂ (pristine) |
| `damaged_Df_C2.csv` | Higuchi fractal dimension for C₂ (damaged) |
| `normalized_yields.png` | Plot of normalized yields |
| `entropy_kl.png` | Shannon entropy and KL divergence plots |
| `thermal_spike.png` | Thermal spike temperature plots |

## Using Your Own Data

Prepare a CSV with columns: `energy_keV`, `sample_type`, `material`, `C1`, `C2`, `C3`, `C4`

```python
from mvp_carbon_fragmentation import load_dataset_from_csv
ds = load_dataset_from_csv("path/to/your/data.csv", cluster_range=(1, 4))
```

## Algorithms

- **Normalized yield:** Nₓ(E) = Yₓ(E) / Σₓ Yₓ(E)
- **Shannon entropy:** H(p) = −Σ pᵢ ln pᵢ
- **KL divergence:** D_KL(p‖q) = Σ pᵢ ln(pᵢ/qᵢ)
- **Thermal spike:** Ts = −ΔE / (k_B · ln R), where R = p(A)/p(B)
- **Fractal dimension:** Higuchi method on p(Cⱼ) vs energy

## Dependencies

- Python 3.8+
- numpy ≥ 1.21
- pandas ≥ 1.3
- matplotlib ≥ 3.5

## License

MIT License — see [LICENSE](LICENSE)
