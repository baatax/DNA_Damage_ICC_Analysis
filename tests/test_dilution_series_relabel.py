import json

import pandas as pd

from dna_damage_parquet_pipeline import (
    DNADamageDataLoader,
    ExperimentConfig,
    format_concentration_um,
    relabel_pseudo_dilution_series,
)


def test_relabel_pseudo_dilution_series_uses_configured_max_dose():
    raw = pd.Series(["DMSO", "0.1uM", "0.3uM", "0.9uM"])
    labels, um, levels = relabel_pseudo_dilution_series(
        raw_labels=raw,
        control_label="DMSO",
        max_dose="90uM",
        dilution_factor=3.0,
    )

    assert labels.tolist() == ["DMSO", "10uM", "30uM", "90uM"]
    assert um.tolist() == [0.0, 10.0, 30.0, 90.0]
    assert pd.isna(levels.iloc[0])
    assert levels.iloc[1:].tolist() == [2.0, 1.0, 0.0]


def test_relabel_pseudo_dilution_series_keeps_raw_if_max_dose_invalid():
    raw = pd.Series(["DMSO", "1uM", "3uM"])
    labels, um, levels = relabel_pseudo_dilution_series(
        raw_labels=raw,
        control_label="DMSO",
        max_dose="",
        dilution_factor=3.0,
    )

    assert labels.tolist() == raw.tolist()
    assert um.tolist() == [0.0, 1.0, 3.0]
    assert levels.isna().all()


def test_data_loader_applies_pseudo_dilution_relabel(tmp_path):
    src = tmp_path / "input.parquet"
    pd.DataFrame(
        {
            "Dilut": ["DMSO", "1uM", "3uM", "9uM", "27uM", "81uM", "243uM", "729uM", "2187uM", "6561uM"],
            "f1": list(range(1, 11)),
        }
    ).to_parquet(src, index=False)

    cfg = {
        "metadata": {"control_label": "DMSO", "dilut_column": "Dilut"},
        "genotypes": {
            "WT": {
                "drugs": {
                    "DrugA": {"path": str(src), "max_dose": "100uM", "dilution_factor": 3.0}
                }
            }
        },
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    loader = DNADamageDataLoader(ExperimentConfig.from_json(cfg_path))
    out = loader.load_single_parquet("WT", "DrugA")

    assert "raw_dilut_string" in out.columns
    assert "dilution_level" in out.columns
    observed = out[["raw_dilut_string", "dilut_string", "dilut_um"]]
    got = {r.raw_dilut_string: (r.dilut_string, r.dilut_um) for r in observed.itertuples(index=False)}
    assert got["DMSO"] == ("DMSO", 0.0)
    assert got["6561uM"] == ("100uM", 100.0)
    assert got["2187uM"] == ("33.3333uM", 100.0 / 3.0)
    assert got["729uM"] == ("11.1111uM", 100.0 / 9.0)


def test_format_concentration_um_scales_units():
    assert format_concentration_um(3000.0) == "3mM"
    assert format_concentration_um(30.0) == "30uM"
    assert format_concentration_um(0.03) == "30nM"
