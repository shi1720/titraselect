# TitraSelect

Reproducible research accompanying **When Can a Sparse Titration Replace a Full Curve? Affine-Invariant Screening with Finite-Library Error Control**, Shivam Gupta, October 2026.

[Read the manuscript](manuscript/paper.pdf). This is an unpublished research manuscript; inclusion here does not imply peer review or acceptance.

## Research question

When can a blank and two antigen concentrations support an affinity-threshold decision, and when should a screening system request a full curve? The implementation combines a one-site-model contrast with exact sampling-without-replacement validation of accepted decisions. An optional second certificate protects reference hits from being discarded as negatives.

The benchmark reuses measured CR6261/CR9114 and CH65 titrations from Phillips et al. (eLife, 2021 and 2023). It contains 840 primary method–split runs (8 methods, 21 source/reference configurations, 5 partitions) and 3,528 sensitivity runs. The separately labeled affine stress experiment is constructed computationally; it is not a new laboratory experiment.

An important failure case: on CH65–SI06 a fixed three-point rule averages 98.7% accepted coverage and 2.1% accepted error, but retains only 29.8% of reference hits. Class-specific error control alone therefore does not ensure discovery utility. The hit-preservation certificate exposes the additional validation cost and can require complete deferral.

## Reproduce

Tested with Python 3.9.6 and the pinned packages in `requirements.txt`. Use a virtual environment. The data download requires about 830 MB; source tables remain under their original terms and are not redistributed in this repository.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python download_data.py --data data
python test_titraselect.py
python validate_data.py --data data --out results/alignment.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python titraselect.py --data data --out results
python run_sensitivity.py --data data
python make_figures.py
cd manuscript
tectonic paper.tex
```

Tectonic is the LaTeX compiler used for the paper. It is installed separately. The benchmark itself requires no GPU, paid API, proprietary model, or Quine access. Primary execution took about 108 seconds on the recorded environment; timing varies by machine. The full workflow also includes downloads, validation, sensitivity runs, figure generation, and compilation.

The independent rerun matched all 168 seed-1720 run records exactly. The tests check the contrast algebra and affine invariance, exhaustive small-population coverage of the hypergeometric bound, and independence of policy selection from held-out outcomes. Data alignment reconstructs published fit quality for 18,475 curves across 21 source tables; the maximum absolute difference is about 5.46e-10.

## Files

- `titraselect.py`: loaders, mechanistic and learned readouts, calibration, evaluation, and constructed stress test.
- `test_titraselect.py`, `validate_data.py`: numerical/statistical and source-column alignment checks.
- `download_data.py`, `provenance/`: commit-pinned URLs, SHA-256 hashes, concentration grids, and software versions.
- `results/`: primary runs, selected gates, cohort descriptions, and compressed held-out prediction records for the first seed.
- `sensitivity/`: calibration-budget, threshold, risk-level, and hit-protection checks, three seeds each.
- `make_figures.py`, `manuscript/`: vector figures, numerical tables, bibliography, LaTeX source, and PDF.

The corrected CH65a table `Kds_CH65a_f3jrem.csv` follows the original authors' browser-cleanup notebook, which removes a spurious 4,096-genotype fragment. Genotypes are joined as strings. Original concentration grids and the CH65 response-column rotation are reconstructed from the authors' analysis metadata. The alignment check is intended to catch column-order mistakes.

## Interpretation and limits

Reference labels come from independent-day published full-curve fits, not latent affinity ground truth. Complete-case cohorts are restricted using reference quality. The guarantees concern the remaining members of a fixed library under randomized calibration sampling and declared reference labels; they do not automatically cover a new library, a changed assay, or biological validity. Repeated partitions are computational robustness checks, not independent biological experiments. Affine invariance covers a shared positive gain and additive offset across concentrations; it does not cover concentration-dependent artifacts. No prospective laboratory savings, therapeutic efficacy, patentability, or universal novelty is established.

The manuscript distinguishes established binding/statistical foundations from the contribution of their integration and the empirical benchmark. Exploratory thresholds and additional hit-retention analyses are explicitly reported.

## Sources and citation

- Phillips et al. (2021), [eLife 10:e71393](https://doi.org/10.7554/eLife.71393); [source repository](https://github.com/klawrence26/bnab-landscapes).
- Phillips et al. (2023), [eLife 12:e83628](https://doi.org/10.7554/eLife.83628); [source repository](https://github.com/amphilli/CH65-comblib).

Please cite the original experimental studies when reusing their measurements. Cite this artifact as: Gupta, S. (2026). *When Can a Sparse Titration Replace a Full Curve? Affine-Invariant Screening with Finite-Library Error Control*. Research manuscript and TitraSelect software. https://github.com/shi1720/titraselect.

## License

Original software in this repository is MIT licensed. The manuscript is copyright Shivam Gupta, 2026. Original experimental datasets and cited publications retain their respective authors' rights and terms.
