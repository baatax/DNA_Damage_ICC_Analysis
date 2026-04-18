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

This executes:
- dataset/config generation tests
- end-to-end smoke test for pipeline plotting/output generation using repo test fixtures
- pseudo-dilution relabeling tests for max-dose driven concentration correction
- dose-response fit plot checks including EC50 as % max concentration and log10(% max)
- QC reporting checks, including presence of a human-readable exclusion report

## Notes

- `PYTHONPATH=.` ensures local modules (for example `run_dna_damage_pipeline.py`) are importable during tests.
- The tests use synthetic + bundled test data and do not require external services.
