import json
import re
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from demo_dna_damage_pipeline import generate_synthetic_dna_damage_data
from dna_damage_plotting import PlotGenerator
from run_dna_damage_pipeline import DNADamageProductionPipeline


class PlotGenerationSmokeTest(unittest.TestCase):
    def test_ec50_plotting_replaces_preexisting_pc_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_root = Path(tmpdir)
            csv_path = out_root / "uncorrected" / "plots" / "ec50_focused" / "ec50_profiles_uncorrected.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)

            # Include pre-existing PC1/PC2 columns that should be ignored by
            # the EC50-focused re-embedding code path.
            df = pd.DataFrame(
                {
                    "genotype": ["WT", "WT", "KO", "KO"],
                    "PC1": [100, 101, -100, -101],
                    "PC2": [50, 51, -50, -51],
                    "feat_a": [1.0, 1.2, 3.4, 3.7],
                    "feat_b": [2.0, 2.1, 4.2, 4.5],
                    "dilut_um": [0.1, 0.1, 0.1, 0.1],
                }
            )
            df.to_csv(csv_path, index=False)

            plotter = PlotGenerator(out_root)
            results = plotter.generate_for_csv(csv_path)

            self.assertTrue(results.plot_paths, "EC50 plotting did not generate output plots")

            # The merged dataframe used for plotting should not keep duplicated
            # PC columns, otherwise matplotlib receives two x/y arrays and plots
            # 2x points at mismatched coordinates.
            pca_result = plotter._compute_pca(df, ["feat_a", "feat_b"])
            self.assertIsNotNone(pca_result)
            scores_df, _, _ = pca_result
            base_df = df.drop(columns=[c for c in ["PC1", "PC2"] if c in df.columns], errors="ignore")
            merged = pd.concat([base_df.reset_index(drop=True), scores_df.reset_index(drop=True)], axis=1)
            self.assertEqual(list(merged.columns).count("PC1"), 1)
            self.assertEqual(list(merged.columns).count("PC2"), 1)

    def test_pipeline_generates_required_plots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            parquet_dir = root / "parquets"
            parquet_dir.mkdir(parents=True, exist_ok=True)

            df = generate_synthetic_dna_damage_data(
                n_cells_per_well=20,
                n_wells_per_dose=1,
                genotype="WT",
                drug="Etoposide",
                random_seed=7,
                n_doses=9,
            )
            pq_path = parquet_dir / "WT_Etoposide.parquet"
            df.to_parquet(pq_path, index=False)

            out_dir = root / "analysis_results"
            config = {
                "metadata": {
                    "experiment_name": "plot_smoke",
                    "date": "2026-01-01",
                    "control_label": "DMSO",
                    "dilut_column": "Dilut",
                    "output_dir": str(out_dir),
                },
                "genotypes": {
                    "WT": {
                        "drugs": {
                            "Etoposide": {
                                "path": str(pq_path),
                                "ec50_um": 1.5,
                                "max_dose": "100uM",
                                "moa": "Topoisomerase II inhibitor",
                            }
                        }
                    }
                },
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

            pipeline = DNADamageProductionPipeline(config_path=config_path, resume=False, n_workers=1)
            pipeline.run()

            timestamped_outputs = sorted(root.glob("analysis_results_*"))
            self.assertTrue(timestamped_outputs, "timestamped output directory was not created")
            run_out_dir = timestamped_outputs[-1]

            log_candidates = list(run_out_dir.glob("pipeline_*.log"))
            self.assertTrue(log_candidates, "pipeline log was not created")
            log_path = log_candidates[-1]
            log_text = log_path.read_text(encoding="utf-8")
            match = re.search(r"Generated\s+(\d+)\s+plots", log_text)
            self.assertIsNotNone(match, "Plot generation step did not report generated plots")
            self.assertGreater(int(match.group(1)), 0, "Plot generation produced zero plots")

            qc_report = run_out_dir / "tables" / "qc_exclusion_report.md"
            self.assertTrue(qc_report.exists(), "Human-readable QC exclusion report was not generated")
            report_text = qc_report.read_text(encoding="utf-8")
            self.assertIn("# QC Exclusion Report", report_text)

            # QC plots should still be at the top level
            qc_plots = list((run_out_dir / "plots" / "qc").glob("*.png"))
            self.assertTrue(qc_plots, "No QC plots generated under plots/qc/")

            # PCA embedding and dose-response plots should exist
            embedding_dir = run_out_dir / "plots" / "embedding"
            dr_dir = run_out_dir / "plots" / "dose_response"
            found_pca = embedding_dir.exists() and list(embedding_dir.glob("*.png"))
            found_dr = dr_dir.exists() and list(dr_dir.glob("*.png"))
            self.assertTrue(
                found_pca or found_dr,
                "No PCA embedding or dose-response plots generated",
            )

            # EC50 fit-specific scatter plots were intentionally removed; EC50 values
            # are now shown in an on-plot table in summary dose-response figures.
            pct_max_plots = list(dr_dir.glob("*_ec50_pct_max.png"))
            pct_max_log_plots = list(dr_dir.glob("*_log10_ec50_pct_max.png"))
            self.assertFalse(pct_max_plots, "Deprecated EC50 % max concentration plots should not be generated")
            self.assertFalse(pct_max_log_plots, "Deprecated log10 EC50 % max concentration plots should not be generated")


if __name__ == "__main__":
    unittest.main()
