import json
import re
import tempfile
import unittest
from pathlib import Path

from demo_dna_damage_pipeline import generate_synthetic_dna_damage_data
from run_dna_damage_pipeline import DNADamageProductionPipeline


class PlotGenerationSmokeTest(unittest.TestCase):
    def test_step10_generates_required_plots(self):
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

            log_path = out_dir / "logs" / "pipeline.log"
            self.assertTrue(log_path.exists(), "pipeline log was not created")
            log_text = log_path.read_text(encoding="utf-8")
            match = re.search(r"Generated\s+(\d+)\s+plots", log_text)
            self.assertIsNotNone(match, "Step 10 did not report generated plots")
            self.assertGreater(int(match.group(1)), 0, "Step 10 generated zero plots")

            self.assertTrue((out_dir / "plots" / "qc" / "well_qc_metrics_distributions.png").exists())
            self.assertTrue((out_dir / "plots" / "qc" / "well_qc_metrics_qc_pass_counts.png").exists())

            dr_fits = list((out_dir / "plots" / "dose_response").glob("dose_response_fits_*_ec50.png"))
            self.assertTrue(dr_fits, "dose-response fit plot not generated")

            pca_modes = ["across_all", "per_genotype"]
            found_pca = any(
                (out_dir / "plots" / mode / f"profiles_pca_{mode}_pc1_pc2.png").exists()
                for mode in pca_modes
            )
            self.assertTrue(found_pca, "PCA PC1/PC2 plot not generated")


if __name__ == "__main__":
    unittest.main()
