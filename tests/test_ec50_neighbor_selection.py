import json
import tempfile
from pathlib import Path

import pandas as pd

from run_dna_damage_pipeline import DNADamageProductionPipeline


def _build_pipeline(tmpdir: str, ec50_um: float) -> DNADamageProductionPipeline:
    cfg = {
        "metadata": {
            "experiment_name": "ec50_neighbors",
            "date": "2026-01-01",
            "control_label": "DMSO",
            "dilut_column": "Dilut",
            "output_dir": str(Path(tmpdir) / "out"),
        },
        "genotypes": {
            "WT": {"drugs": {"DrugA": {"path": str(Path(tmpdir) / "dummy.parquet"), "ec50_um": ec50_um}}},
            "KO": {"drugs": {"DrugA": {"path": str(Path(tmpdir) / "dummy.parquet"), "ec50_um": ec50_um}}},
        },
    }
    cfg_path = Path(tmpdir) / "cfg.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    return DNADamageProductionPipeline(config_path=cfg_path, resume=False, n_workers=1)


def test_collect_ec50_neighbor_rows_selects_lower_ec50_higher():
    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline = _build_pipeline(tmpdir, ec50_um=1.0)
        rows = []
        for geno in ["WT", "KO"]:
            for dose in [0.3, 1.0, 3.0]:
                rows.append({"genotype": geno, "drug": "DrugA", "dilut_um": dose, "value": dose})
        profiles = pd.DataFrame(rows)
        out = pipeline._collect_ec50_neighbor_rows(profiles)

        assert not out.empty
        counts = out.groupby(["genotype", "ec50_band"])["dilut_um"].nunique().to_dict()
        assert counts[("WT", "lower")] == 1
        assert counts[("WT", "ec50")] == 1
        assert counts[("WT", "higher")] == 1


def test_collect_ec50_neighbor_rows_caps_higher_at_max_dose():
    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline = _build_pipeline(tmpdir, ec50_um=9.0)
        profiles = pd.DataFrame(
            [
                {"genotype": "WT", "drug": "DrugA", "dilut_um": 1.0, "value": 1.0},
                {"genotype": "WT", "drug": "DrugA", "dilut_um": 3.0, "value": 3.0},
                {"genotype": "WT", "drug": "DrugA", "dilut_um": 9.0, "value": 9.0},
            ]
        )
        out = pipeline._collect_ec50_neighbor_rows(profiles)

        high = out[out["ec50_band"] == "higher"]
        # When EC50 is at the highest available dose, "higher" resolves to EC50.
        assert set(high["dilut_um"].tolist()) == {9.0}
