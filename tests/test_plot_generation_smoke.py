import json
import re
import tempfile
import unittest
from pathlib import Path

from demo_dna_damage_pipeline import generate_synthetic_dna_damage_data
from run_dna_damage_pipeline import DNADamageProductionPipeline


class PlotGenerationSmokeTest(unittest.TestCase):
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
                n_doses=5,
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

            pct_max_plots = list(dr_dir.glob("*_ec50_pct_max.png"))
            pct_max_log_plots = list(dr_dir.glob("*_log10_ec50_pct_max.png"))
            self.assertTrue(pct_max_plots, "No EC50 % max concentration plots generated")
            self.assertTrue(pct_max_log_plots, "No log10 EC50 % max concentration plots generated")


if __name__ == "__main__":
    unittest.main()
