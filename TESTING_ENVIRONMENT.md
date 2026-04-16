# Testing Environment Setup

This project is validated with Python **3.10+** and the bundled dataset under `test_data/`.

## Required packages

Install the core runtime + test dependencies:

```bash
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

## Notes

- `PYTHONPATH=.` ensures local modules (for example `run_dna_damage_pipeline.py`) are importable during tests.
- The tests use synthetic + bundled test data and do not require external services.
