# Testing Environment Setup

This project is validated with Python **3.10+** and repository-provided test fixtures (synthetic fixtures generated in tests plus config-driven test data in-repo when available).

## Required packages

Create an isolated environment and install the core runtime + test dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install pandas numpy scipy matplotlib scikit-learn pyarrow pytest
```

## Run tests

From repository root:

```bash
PYTHONPATH=. pytest -q
```

To specifically validate plotting behavior related to EC50-focused PCA and
avoid regressions where stale `PC1`/`PC2` columns can duplicate plotted points:

```bash
PYTHONPATH=. pytest -q tests/test_plot_generation_smoke.py -k ec50
```

This executes:
- dataset/config generation tests
- end-to-end smoke test for pipeline plotting/output generation using repo test fixtures
- pseudo-dilution relabeling tests for max-dose driven concentration correction
- dose-response fit plot checks including EC50 as % max concentration and log10(% max)
- QC reporting checks, including presence of a human-readable exclusion report

## Run pipeline against bundled repository test data

The repository includes parquet fixtures under `test_data/`. To run the full
pipeline using those fixtures:

```bash
python run_dna_damage_pipeline.py Pilot_1_config_test_config.json --output ./test_output_pilot1
```

This is useful for integration checks beyond unit tests, including validating
EC50 subset generation and downstream EC50 PCA plots.

## Notes

- `PYTHONPATH=.` ensures local modules (for example `run_dna_damage_pipeline.py`) are importable during tests.
- The tests use synthetic + bundled test data and do not require external services.
