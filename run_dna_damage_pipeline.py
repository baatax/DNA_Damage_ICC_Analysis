#!/usr/bin/env python3
"""
DNA Damage Panel Analysis Pipeline - Production Version
========================================================

This script is designed for production use on SLURM clusters.

Usage:
    # Local execution
    python run_dna_damage_pipeline.py config.json --output ./results

    # SLURM submission
    sbatch slurm_submit.sh config.json

Features:
    - Configurable via JSON
    - SLURM-compatible (no interactive prompts)
    - Comprehensive logging
    - Checkpoint/resume support
    - Memory-efficient processing
    - Parallel processing with configurable workers
    - Crowding correction (corrected and uncorrected analyses)
    - PCA with feature contribution plots
    - Dose-correlated feature analysis at EC50
    - Publication-quality plot generation

Author: DNA Damage Analysis Pipeline
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Suppress warnings for cleaner logs
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(
    output_dir: Path,
    log_level: str = "INFO",
    log_to_console: bool = True,
) -> logging.Logger:
    """Setup logging to both file and console."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Create logger
    logger = logging.getLogger("DNADamagePipeline")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    logger.handlers = []

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_formatter = logging.Formatter('%(levelname)-8s | %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger


# =============================================================================
# IMPORT PIPELINE COMPONENTS
# =============================================================================

# Try to import from installed package or local file
try:
    from dna_damage_parquet_pipeline import (
        ExperimentConfig,
        DNADamageDataLoader,
        DNADamagePreprocessor,
        CrowdingAnalyzer,
        QCMetricsCompiler,
        FeatureSelector,
        parse_dilution_string,
    )
    from dose_response_analysis import (
        DoseResponseAnalyzer,
        StatisticalComparator,
        ResponseSummarizer,
        analyze_dose_response,
    )
except ImportError as e:
    print(f"ERROR: Could not import pipeline modules: {e}")
    print("Make sure dna_damage_parquet_pipeline.py and dose_response_analysis.py")
    print("are in the same directory or in your PYTHONPATH.")
    sys.exit(1)

# Plotting (optional — degrades gracefully)
try:
    from pipeline_plots import (
        plot_pca_scatter,
        plot_pca_feature_contributions,
        plot_pca_variance_explained,
        plot_crowding_excluded_features,
        plot_dose_feature_lines,
        plot_dose_correlation_ranking,
        HAS_MATPLOTLIB,
    )
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# CHECKPOINT MANAGER
# =============================================================================

class CheckpointManager:
    """Manage pipeline checkpoints for resume capability."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.checkpoint_file = self.output_dir / ".pipeline_checkpoint.json"
        self.checkpoints: Dict[str, Any] = {}
        self._load()

    def _load(self):
        """Load existing checkpoints."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    self.checkpoints = json.load(f)
            except Exception:
                self.checkpoints = {}

    def _save(self):
        """Save checkpoints to disk."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.checkpoints, f, indent=2)

    def is_complete(self, step: str) -> bool:
        """Check if a step is already complete."""
        return self.checkpoints.get(step, {}).get("complete", False)

    def mark_complete(self, step: str, metadata: Optional[Dict] = None):
        """Mark a step as complete."""
        self.checkpoints[step] = {
            "complete": True,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        self._save()

    def get_metadata(self, step: str) -> Dict:
        """Get metadata for a step."""
        return self.checkpoints.get(step, {}).get("metadata", {})

    def clear(self):
        """Clear all checkpoints."""
        self.checkpoints = {}
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()


# =============================================================================
# MAIN PIPELINE CLASS
# =============================================================================

class DNADamageProductionPipeline:
    """
    Production-ready DNA damage analysis pipeline.

    Designed for SLURM cluster execution with:
        - Checkpoint/resume support
        - Comprehensive logging
        - Memory-efficient processing
        - Configurable parallelization
        - Crowding-corrected and uncorrected parallel analyses
        - PCA plots with feature contributions
        - Dose-correlated feature analysis at EC50
    """

    def __init__(
        self,
        config_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        n_workers: int = 4,
        resume: bool = True,
        log_level: str = "INFO",
        crowding_threshold: float = 0.3,
    ):
        """
        Initialize the pipeline.

        Parameters
        ----------
        config_path : str or Path
            Path to JSON configuration file
        output_dir : str or Path, optional
            Output directory (overrides config)
        n_workers : int
            Number of parallel workers
        resume : bool
            Whether to resume from checkpoints
        log_level : str
            Logging level (DEBUG, INFO, WARNING, ERROR)
        crowding_threshold : float
            |Spearman r| threshold above which features are excluded
            in the crowding-corrected analysis (default 0.3)
        """
        self.config_path = Path(config_path)
        self.n_workers = n_workers
        self.resume = resume
        self.crowding_threshold = crowding_threshold

        # Load configuration
        self.config = ExperimentConfig.from_json(config_path)

        # Set output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        elif self.config.output_dir:
            self.output_dir = Path(self.config.output_dir)
        else:
            self.output_dir = Path(f"./output_{self.config.experiment_name}")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = setup_logging(self.output_dir, log_level)

        # Setup checkpoint manager
        self.checkpoint = CheckpointManager(self.output_dir)
        if not resume:
            self.checkpoint.clear()

        # Initialize components
        self.loader = DNADamageDataLoader(self.config)
        self.preprocessor_per_geno = DNADamagePreprocessor(
            norm_method="robust_zscore",
            batch_col="plate",
        )
        self.preprocessor_across = DNADamagePreprocessor(
            norm_method="robust_zscore",
            batch_col="plate",
        )
        self.crowding_analyzer = CrowdingAnalyzer(
            dilut_column=self.config.dilut_column,
            control_label=self.config.control_label,
        )
        self.qc_compiler = QCMetricsCompiler()
        self.feature_selector = FeatureSelector()
        self.dose_response_analyzer = DoseResponseAnalyzer(
            dose_column='dilut_um',
            control_label=self.config.control_label,
        )
        self.response_summarizer = ResponseSummarizer(
            dose_col='dilut_um',
            control_label=self.config.control_label,
        )
        self.statistical_comparator = StatisticalComparator()

        # Data storage
        self.raw_data: Optional[pd.DataFrame] = None
        self.data_per_geno: Optional[pd.DataFrame] = None
        self.data_across_all: Optional[pd.DataFrame] = None
        self.feature_cols_per_geno: List[str] = []
        self.feature_cols_across: List[str] = []

        # Crowding correction
        self.feature_cols_per_geno_corrected: List[str] = []
        self.feature_cols_across_corrected: List[str] = []
        self.crowding_excluded_features: List[str] = []
        self.crowding_correlations: Optional[pd.Series] = None

        # EC50 lookup  {(genotype, drug): ec50_um}
        self.ec50_lookup: Dict[Tuple[str, str], float] = {}

        # Track timing
        self.timing: Dict[str, float] = {}

    # -----------------------------------------------------------------
    # Directory setup
    # -----------------------------------------------------------------
    def _setup_directories(self):
        """Create output directory structure."""
        dirs = [
            self.output_dir,
            self.output_dir / "per_genotype",
            self.output_dir / "across_all",
            self.output_dir / "crowding",
            self.output_dir / "qc",
            self.output_dir / "per_well",
            self.output_dir / "per_well" / "per_genotype",
            self.output_dir / "per_well" / "across_all",
            self.output_dir / "pca",
            self.output_dir / "dose_response",
            self.output_dir / "dose_correlated",
            self.output_dir / "statistics",
            self.output_dir / "plots",
            self.output_dir / "plots" / "pca",
            self.output_dir / "plots" / "crowding",
            self.output_dir / "plots" / "dose_correlated",
        ]
        # PCA sub-dirs: mode / correction
        for mode in ("per_genotype", "across_all"):
            for corr in ("uncorrected", "corrected"):
                dirs.append(self.output_dir / "pca" / mode / corr)
                dirs.append(self.output_dir / "plots" / "pca" / mode / corr)
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------
    def _log_step(self, step: str, message: str):
        self.logger.info(f"[{step}] {message}")

    def _log_timing(self, step: str, duration: float):
        self.timing[step] = duration
        self.logger.info(f"[{step}] Completed in {duration:.2f} seconds")

    def _build_ec50_lookup(self):
        """Build EC50 lookup from config."""
        for geno_name, geno_config in self.config.genotypes.items():
            for drug_name, drug_config in geno_config.drugs.items():
                if drug_config.ec50_um is not None:
                    self.ec50_lookup[(geno_name, drug_name)] = drug_config.ec50_um

    # =================================================================
    # Pipeline runner
    # =================================================================
    def run(self) -> Dict[str, Path]:
        """Run the complete analysis pipeline."""
        start_time = time.time()

        self.logger.info("=" * 70)
        self.logger.info("DNA DAMAGE PANEL ANALYSIS PIPELINE")
        self.logger.info("=" * 70)
        self.logger.info(f"Experiment: {self.config.experiment_name}")
        self.logger.info(f"Config: {self.config_path}")
        self.logger.info(f"Output: {self.output_dir}")
        self.logger.info(f"Workers: {self.n_workers}")
        self.logger.info(f"Resume: {self.resume}")
        self.logger.info(f"Crowding threshold: {self.crowding_threshold}")
        self.logger.info("=" * 70)

        self._setup_directories()
        self._build_ec50_lookup()
        output_files: Dict[str, Any] = {}

        try:
            # Step 1: Load data
            output_files.update(self._step_load_data())

            # Step 2: Preprocess per-genotype
            output_files.update(self._step_preprocess_per_genotype())

            # Step 3: Preprocess across-all
            output_files.update(self._step_preprocess_across_all())

            # Step 4: Compile crowding metrics
            output_files.update(self._step_compile_crowding())

            # Step 5: Crowding correction analysis
            output_files.update(self._step_crowding_correction())

            # Step 6: Compile QC metrics
            output_files.update(self._step_compile_qc())

            # Step 7: Generate per-well CSVs
            output_files.update(self._step_generate_per_well())

            # Step 8: Compute PCA (uncorrected + corrected)
            output_files.update(self._step_compute_pca())

            # Step 9: Dose-response analysis
            output_files.update(self._step_dose_response_analysis())

            # Step 10: Dose-correlated feature analysis
            output_files.update(self._step_dose_correlated_analysis())

            # Step 11: Statistical comparisons
            output_files.update(self._step_statistical_comparisons())

            # Step 12: Generate plots
            output_files.update(self._step_generate_plots())

            # Step 13: Save manifest
            self._save_manifest(output_files)

            total_time = time.time() - start_time
            self.logger.info("=" * 70)
            self.logger.info("PIPELINE COMPLETE")
            self.logger.info(f"Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
            self.logger.info(f"Results: {self.output_dir}")
            self.logger.info("=" * 70)

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            self.logger.error(traceback.format_exc())
            raise

        return output_files

    # =================================================================
    # Step 1 – Load data
    # =================================================================
    def _step_load_data(self) -> Dict[str, Path]:
        """Step 1: Load all parquet data."""
        step = "1_load_data"

        if self.resume and self.checkpoint.is_complete(step):
            self._log_step(step, "Loading from checkpoint...")
            cache_path = self.output_dir / "cache" / "raw_data.parquet"
            if cache_path.exists():
                self.raw_data = pd.read_parquet(cache_path)
                self._log_step(step, f"Loaded {len(self.raw_data):,} cells from cache")
                return {}

        self._log_step(step, "Loading parquet files...")
        start = time.time()

        self.raw_data = self.loader.load_all()

        if self.raw_data.empty:
            raise ValueError("No data loaded. Check file paths in configuration.")

        # Cache raw data for resume
        cache_dir = self.output_dir / "cache"
        cache_dir.mkdir(exist_ok=True)
        self.raw_data.to_parquet(cache_dir / "raw_data.parquet", index=False)

        self._log_timing(step, time.time() - start)
        self._log_step(step, f"Loaded {len(self.raw_data):,} cells")
        self._log_step(step, f"Genotypes: {self.raw_data['genotype'].unique().tolist()}")
        self._log_step(step, f"Drugs: {self.raw_data['drug'].unique().tolist()}")

        self.checkpoint.mark_complete(step, {
            "n_cells": len(self.raw_data),
            "genotypes": self.raw_data['genotype'].unique().tolist(),
            "drugs": self.raw_data['drug'].unique().tolist(),
        })

        return {}

    # =================================================================
    # Step 2 – Preprocess per-genotype
    # =================================================================
    def _step_preprocess_per_genotype(self) -> Dict[str, Path]:
        step = "2_preprocess_per_genotype"

        if self.resume and self.checkpoint.is_complete(step):
            self._log_step(step, "Loading from checkpoint...")
            cache_path = self.output_dir / "cache" / "data_per_geno.parquet"
            if cache_path.exists():
                self.data_per_geno = pd.read_parquet(cache_path)
                self.feature_cols_per_geno = self.checkpoint.get_metadata(step).get("feature_cols", [])
                return {}

        self._log_step(step, "Preprocessing (per-genotype normalization)...")
        start = time.time()

        self.data_per_geno, self.feature_cols_per_geno = (
            self.preprocessor_per_geno.preprocess_per_genotype(self.raw_data)
        )

        cache_dir = self.output_dir / "cache"
        self.data_per_geno.to_parquet(cache_dir / "data_per_geno.parquet", index=False)

        self._log_timing(step, time.time() - start)
        self._log_step(step, f"Feature columns: {len(self.feature_cols_per_geno)}")

        self.checkpoint.mark_complete(step, {"feature_cols": self.feature_cols_per_geno})
        return {}

    # =================================================================
    # Step 3 – Preprocess across-all
    # =================================================================
    def _step_preprocess_across_all(self) -> Dict[str, Path]:
        step = "3_preprocess_across_all"

        if self.resume and self.checkpoint.is_complete(step):
            self._log_step(step, "Loading from checkpoint...")
            cache_path = self.output_dir / "cache" / "data_across_all.parquet"
            if cache_path.exists():
                self.data_across_all = pd.read_parquet(cache_path)
                self.feature_cols_across = self.checkpoint.get_metadata(step).get("feature_cols", [])
                return {}

        self._log_step(step, "Preprocessing (across-all normalization)...")
        start = time.time()

        self.data_across_all, self.feature_cols_across = (
            self.preprocessor_across.preprocess_across_all(self.raw_data)
        )

        cache_dir = self.output_dir / "cache"
        self.data_across_all.to_parquet(cache_dir / "data_across_all.parquet", index=False)

        self._log_timing(step, time.time() - start)
        self._log_step(step, f"Feature columns: {len(self.feature_cols_across)}")

        self.checkpoint.mark_complete(step, {"feature_cols": self.feature_cols_across})
        return {}

    # =================================================================
    # Step 4 – Compile crowding metrics
    # =================================================================
    def _step_compile_crowding(self) -> Dict[str, Path]:
        step = "4_compile_crowding"
        output_files: Dict[str, Path] = {}

        if self.resume and self.checkpoint.is_complete(step):
            self._log_step(step, "Skipping (already complete)")
            return {}

        self._log_step(step, "Compiling crowding metrics...")
        start = time.time()

        crowding_per_geno = self.crowding_analyzer.compile_crowding_by_drug_dose(
            self.data_per_geno, "per_genotype"
        )
        crowding_across = self.crowding_analyzer.compile_crowding_by_drug_dose(
            self.data_across_all, "across_all"
        )
        crowding_combined = pd.concat(
            [crowding_per_geno, crowding_across], ignore_index=True
        )

        crowding_path = self.output_dir / "crowding" / "crowding_by_drug_dose.csv"
        crowding_combined.to_csv(crowding_path, index=False)
        output_files['crowding'] = crowding_path

        if not crowding_per_geno.empty:
            p = self.output_dir / "crowding" / "crowding_per_genotype.csv"
            crowding_per_geno.to_csv(p, index=False)
            output_files['crowding_per_genotype'] = p
        if not crowding_across.empty:
            p = self.output_dir / "crowding" / "crowding_across_all.csv"
            crowding_across.to_csv(p, index=False)
            output_files['crowding_across_all'] = p

        self._log_timing(step, time.time() - start)
        self._log_step(step, f"Saved: {crowding_path}")
        self.checkpoint.mark_complete(step)
        return output_files

    # =================================================================
    # Step 5 – Crowding correction analysis  (NEW)
    # =================================================================
    def _step_crowding_correction(self) -> Dict[str, Path]:
        """Identify features correlated with cell crowding.

        Computes Spearman correlation of every feature with the primary
        crowding metric across the combined dataset.  Features above
        ``self.crowding_threshold`` are excluded from the *corrected*
        feature set used in downstream PCA / statistics.
        """
        step = "5_crowding_correction"
        output_files: Dict[str, Path] = {}

        if self.resume and self.checkpoint.is_complete(step):
            self._log_step(step, "Loading from checkpoint...")
            meta = self.checkpoint.get_metadata(step)
            self.crowding_excluded_features = meta.get("excluded_features", [])
            excluded_set = set(self.crowding_excluded_features)
            self.feature_cols_per_geno_corrected = [
                f for f in self.feature_cols_per_geno if f not in excluded_set
            ]
            self.feature_cols_across_corrected = [
                f for f in self.feature_cols_across if f not in excluded_set
            ]
            # Reload correlations for plotting
            corr_path = self.output_dir / "crowding" / "crowding_feature_correlations.csv"
            if corr_path.exists():
                corr_df = pd.read_csv(corr_path)
                self.crowding_correlations = pd.Series(
                    corr_df['spearman_r'].values, index=corr_df['feature'].values
                )
            return {}

        self._log_step(step, "Computing crowding-correction analysis...")
        start = time.time()

        # Identify primary crowding metric
        crowding_candidates = [
            'nn_dist_px', 'crowding_local_mean', 'nbr_count_r100',
            'nbr_count_r50', 'mean_k3_dist_px',
        ]
        crowding_metric = None
        for c in crowding_candidates:
            if c in self.raw_data.columns:
                crowding_metric = c
                break

        if crowding_metric is None:
            self._log_step(step, "No crowding metric found; skipping correction")
            self.crowding_excluded_features = []
            self.feature_cols_per_geno_corrected = list(self.feature_cols_per_geno)
            self.feature_cols_across_corrected = list(self.feature_cols_across)
            self.checkpoint.mark_complete(step, {"excluded_features": []})
            return output_files

        self._log_step(step, f"Primary crowding metric: {crowding_metric}")

        # All unique features across both normalization modes
        all_features = sorted(set(self.feature_cols_per_geno) | set(self.feature_cols_across))
        # Exclude crowding columns themselves
        crowding_cols_set = set(CrowdingAnalyzer.CROWDING_COLS)
        analysis_features = [f for f in all_features if f not in crowding_cols_set]

        # Compute Spearman correlations across ALL combined raw data
        correlations: Dict[str, float] = {}
        for feat in analysis_features:
            if feat not in self.raw_data.columns:
                continue
            mask = self.raw_data[feat].notna() & self.raw_data[crowding_metric].notna()
            if mask.sum() < 20:
                continue
            try:
                r, _ = spearmanr(
                    self.raw_data.loc[mask, feat].astype(float),
                    self.raw_data.loc[mask, crowding_metric].astype(float),
                )
                if np.isfinite(r):
                    correlations[feat] = r
            except Exception:
                continue

        self.crowding_correlations = pd.Series(correlations)

        # Exclude features above threshold
        excluded = [f for f, r in correlations.items() if abs(r) > self.crowding_threshold]
        self.crowding_excluded_features = excluded
        excluded_set = set(excluded)

        self.feature_cols_per_geno_corrected = [
            f for f in self.feature_cols_per_geno if f not in excluded_set
        ]
        self.feature_cols_across_corrected = [
            f for f in self.feature_cols_across if f not in excluded_set
        ]

        self._log_step(
            step,
            f"Features analysed: {len(correlations)}, excluded: {len(excluded)} "
            f"(|r| > {self.crowding_threshold})",
        )
        self._log_step(step, f"Corrected per-geno features: {len(self.feature_cols_per_geno_corrected)}")
        self._log_step(step, f"Corrected across-all features: {len(self.feature_cols_across_corrected)}")

        # Save correlation table
        corr_df = pd.DataFrame({
            'feature': list(correlations.keys()),
            'spearman_r': list(correlations.values()),
            'abs_r': [abs(r) for r in correlations.values()],
            'excluded': [f in excluded_set for f in correlations.keys()],
        }).sort_values('abs_r', ascending=False)
        corr_path = self.output_dir / "crowding" / "crowding_feature_correlations.csv"
        corr_df.to_csv(corr_path, index=False)
        output_files['crowding_correlations'] = corr_path

        self._log_timing(step, time.time() - start)
        self.checkpoint.mark_complete(step, {"excluded_features": excluded})
        return output_files

    # =================================================================
    # Step 6 – Compile QC metrics
    # =================================================================
    def _step_compile_qc(self) -> Dict[str, Path]:
        step = "6_compile_qc"
        output_files: Dict[str, Path] = {}

        if self.resume and self.checkpoint.is_complete(step):
            self._log_step(step, "Skipping (already complete)")
            return {}

        self._log_step(step, "Compiling QC metrics...")
        start = time.time()

        groupby_cols = ['plate', 'well']
        groupby_cols = [c for c in groupby_cols if c in self.raw_data.columns]

        qc_metrics = self.qc_compiler.compile_well_qc(self.raw_data, groupby_cols)
        qc_path = self.output_dir / "qc" / "well_qc_metrics.csv"
        qc_metrics.to_csv(qc_path, index=False)
        output_files['qc'] = qc_path

        self._log_timing(step, time.time() - start)
        self._log_step(step, f"Saved: {qc_path}")
        self.checkpoint.mark_complete(step)
        return output_files

    # =================================================================
    # Step 7 – Generate per-well CSVs
    # =================================================================
    def _step_generate_per_well(self) -> Dict[str, Path]:
        step = "7_generate_per_well"
        output_files: Dict[str, Path] = {}

        if self.resume and self.checkpoint.is_complete(step):
            self._log_step(step, "Skipping (already complete)")
            return {}

        self._log_step(step, "Generating per-well CSVs...")
        start = time.time()

        for mode, df, feature_cols in [
            ("per_genotype", self.data_per_geno, self.feature_cols_per_geno),
            ("across_all", self.data_across_all, self.feature_cols_across),
        ]:
            mode_dir = self.output_dir / "per_well" / mode
            mode_dir.mkdir(parents=True, exist_ok=True)

            groupby_cols = ['plate', 'well']
            meta_cols = ['genotype', 'drug', 'dilut_string', 'dilut_um',
                         'is_control', 'moa', 'ec50_um']
            groupby_cols = [c for c in groupby_cols if c in df.columns]
            meta_cols = [c for c in meta_cols if c in df.columns]

            if not groupby_cols:
                continue

            agg_dict = {col: 'median' for col in feature_cols if col in df.columns}
            for col in meta_cols:
                agg_dict[col] = 'first'

            well_data = df.groupby(groupby_cols).agg(agg_dict).reset_index()
            cell_counts = df.groupby(groupby_cols).size().rename('cell_count')
            well_data = well_data.merge(cell_counts.reset_index(), on=groupby_cols)

            combined_path = mode_dir / f"well_profiles_{mode}.csv"
            well_data.to_csv(combined_path, index=False)
            output_files[f"well_profiles_{mode}"] = combined_path

            if 'genotype' in well_data.columns:
                for geno in well_data['genotype'].unique():
                    geno_data = well_data[well_data['genotype'] == geno]
                    geno_safe = geno.replace("/", "_").replace("\\", "_")
                    geno_path = mode_dir / f"well_profiles_{geno_safe}_{mode}.csv"
                    geno_data.to_csv(geno_path, index=False)
                    output_files[f"well_profiles_{geno_safe}_{mode}"] = geno_path

            self._log_step(step, f"Saved {mode} well profiles: {len(well_data)} wells")

        self._log_timing(step, time.time() - start)
        self.checkpoint.mark_complete(step)
        return output_files

    # =================================================================
    # Step 8 – PCA (uncorrected + corrected)  (MODIFIED)
    # =================================================================
    def _compute_pca_single(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        out_dir: Path,
        label: str,
    ) -> Dict[str, Path]:
        """Run PCA on *feature_cols*, save CSVs, return output paths."""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        output_files: Dict[str, Path] = {}

        selected = self.feature_selector.select_features(df, feature_cols)
        if len(selected) < 2:
            self._log_step("pca", f"  {label}: <2 features after selection, skipping")
            return output_files

        groupby_cols = ['plate', 'well']
        meta_cols = ['genotype', 'drug', 'dilut_string', 'dilut_um',
                     'is_control', 'moa', 'ec50_um']
        groupby_cols = [c for c in groupby_cols if c in df.columns]
        meta_cols = [c for c in meta_cols if c in df.columns]
        if not groupby_cols:
            return output_files

        agg_dict = {col: 'median' for col in selected}
        for col in meta_cols:
            agg_dict[col] = 'first'

        profiles = df.groupby(groupby_cols).agg(agg_dict).reset_index()
        cell_counts = df.groupby(groupby_cols).size().rename('cell_count')
        profiles = profiles.merge(cell_counts.reset_index(), on=groupby_cols)

        X = profiles[selected].fillna(0).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        n_components = min(10, len(selected), max(len(profiles) - 1, 2))
        if n_components < 2:
            n_components = 2
        pca = PCA(n_components=n_components, random_state=42)
        pcs = pca.fit_transform(X_scaled)

        for i in range(n_components):
            profiles[f'PC{i+1}'] = pcs[:, i]

        # Save profiles
        profiles_path = out_dir / f"profiles_pca_{label}.csv"
        profiles.to_csv(profiles_path, index=False)
        output_files[f"profiles_pca_{label}"] = profiles_path

        # Save loadings
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=selected,
        )
        loadings_path = out_dir / f"pca_loadings_{label}.csv"
        loadings.to_csv(loadings_path)
        output_files[f"pca_loadings_{label}"] = loadings_path

        # Variance explained
        var_explained = pd.DataFrame({
            'PC': [f'PC{i+1}' for i in range(n_components)],
            'variance_explained': pca.explained_variance_,
            'variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_),
        })
        var_path = out_dir / f"pca_variance_{label}.csv"
        var_explained.to_csv(var_path, index=False)
        output_files[f"pca_variance_{label}"] = var_path

        self._log_step(
            "pca",
            f"  {label}: {len(selected)} features, "
            f"PC1={pca.explained_variance_ratio_[0]*100:.1f}%, "
            f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%",
        )
        return output_files

    def _step_compute_pca(self) -> Dict[str, Path]:
        """Step 8: PCA for both corrected and uncorrected feature sets."""
        step = "8_compute_pca"
        output_files: Dict[str, Path] = {}

        if self.resume and self.checkpoint.is_complete(step):
            self._log_step(step, "Skipping (already complete)")
            return {}

        self._log_step(step, "Computing PCA (uncorrected + corrected)...")
        start = time.time()

        combos = [
            ("per_genotype", self.data_per_geno, {
                "uncorrected": self.feature_cols_per_geno,
                "corrected": self.feature_cols_per_geno_corrected,
            }),
            ("across_all", self.data_across_all, {
                "uncorrected": self.feature_cols_across,
                "corrected": self.feature_cols_across_corrected,
            }),
        ]

        for mode, df, feat_sets in combos:
            for correction, feature_cols in feat_sets.items():
                out_dir = self.output_dir / "pca" / mode / correction
                out_dir.mkdir(parents=True, exist_ok=True)
                label = f"{mode}_{correction}"
                output_files.update(
                    self._compute_pca_single(df, feature_cols, out_dir, label)
                )

        self._log_timing(step, time.time() - start)
        self.checkpoint.mark_complete(step)
        return output_files

    # =================================================================
    # Step 9 – Dose-response analysis
    # =================================================================
    def _step_dose_response_analysis(self) -> Dict[str, Path]:
        step = "9_dose_response"
        output_files: Dict[str, Path] = {}

        if self.resume and self.checkpoint.is_complete(step):
            self._log_step(step, "Skipping (already complete)")
            return {}

        self._log_step(step, "Performing dose-response analysis...")
        start = time.time()

        dr_dir = self.output_dir / "dose_response"

        response_cols = [
            'foci_count', 'foci_mean_per_cell', 'ki67_positive',
            'gamma_h2ax_mean_intensity', 'area', 'dapi_mean_intensity',
        ]
        response_cols = [c for c in response_cols if c in self.raw_data.columns]

        if not response_cols:
            self._log_step(step, "Warning: No response columns found")
            self.checkpoint.mark_complete(step)
            return output_files

        # Save EC50 values from config alongside fits
        ec50_df = pd.DataFrame([
            {'genotype': g, 'drug': d, 'config_ec50_um': v}
            for (g, d), v in self.ec50_lookup.items()
        ])
        if not ec50_df.empty:
            ec50_path = dr_dir / "config_ec50_values.csv"
            ec50_df.to_csv(ec50_path, index=False)
            output_files['config_ec50'] = ec50_path

        # Fit dose-response curves
        all_fits = []
        for col in response_cols:
            try:
                fits = self.dose_response_analyzer.fit_drug_response(
                    self.raw_data, col, groupby=['genotype', 'drug']
                )
                if not fits.empty:
                    all_fits.append(fits)
            except Exception as e:
                self._log_step(step, f"Warning: Could not fit {col}: {e}")

        if all_fits:
            fits_df = pd.concat(all_fits, ignore_index=True)
            fits_path = dr_dir / "dose_response_fits.csv"
            fits_df.to_csv(fits_path, index=False)
            output_files['dose_response_fits'] = fits_path
            self._log_step(step, f"Saved fits: {fits_path}")

        # Summary by dose
        summary = self.response_summarizer.summarize_by_dose(
            self.raw_data, response_cols, groupby=['genotype', 'drug']
        )
        if not summary.empty:
            summary_path = dr_dir / "dose_response_summary.csv"
            summary.to_csv(summary_path, index=False)
            output_files['dose_response_summary'] = summary_path

        # Fold-change
        fold_change = self.response_summarizer.compute_fold_change_vs_control(
            self.raw_data, response_cols, groupby=['genotype', 'drug']
        )
        if not fold_change.empty:
            fc_path = dr_dir / "fold_change_vs_control.csv"
            fold_change.to_csv(fc_path, index=False)
            output_files['fold_change'] = fc_path

        self._log_timing(step, time.time() - start)
        self.checkpoint.mark_complete(step)
        return output_files

    # =================================================================
    # Step 10 – Dose-correlated feature analysis  (NEW)
    # =================================================================
    def _step_dose_correlated_analysis(self) -> Dict[str, Path]:
        """Identify features most correlated with drug dose, run PCA at
        each group's respective EC50, and produce line plots.
        """
        step = "10_dose_correlated"
        output_files: Dict[str, Path] = {}

        if self.resume and self.checkpoint.is_complete(step):
            self._log_step(step, "Skipping (already complete)")
            return {}

        self._log_step(step, "Dose-correlated feature analysis...")
        start = time.time()

        dc_dir = self.output_dir / "dose_correlated"
        dc_dir.mkdir(parents=True, exist_ok=True)

        # Use across-all normalised data for cross-group comparisons
        df = self.data_across_all
        feature_cols = self.feature_cols_across

        if df is None or df.empty or 'dilut_um' not in df.columns:
            self._log_step(step, "Warning: insufficient data, skipping")
            self.checkpoint.mark_complete(step)
            return output_files

        # --- 1.  Feature–dose correlations ---
        dose_corrs: Dict[str, float] = {}
        for feat in feature_cols:
            if feat not in df.columns:
                continue
            mask = df[feat].notna() & df['dilut_um'].notna() & (df['dilut_um'] > 0)
            if mask.sum() < 20:
                continue
            try:
                r, _ = spearmanr(
                    df.loc[mask, feat].astype(float),
                    df.loc[mask, 'dilut_um'].astype(float),
                )
                if np.isfinite(r):
                    dose_corrs[feat] = r
            except Exception:
                continue

        if not dose_corrs:
            self._log_step(step, "Warning: no dose-correlated features found")
            self.checkpoint.mark_complete(step)
            return output_files

        dose_corr_series = pd.Series(dose_corrs)
        n_top = min(15, len(dose_corr_series))
        top_features = dose_corr_series.abs().nlargest(n_top).index.tolist()
        self._log_step(step, f"Top {n_top} dose-correlated features identified")

        # Save correlation table
        dose_corr_df = pd.DataFrame({
            'feature': dose_corr_series.index,
            'spearman_r_with_dose': dose_corr_series.values,
        }).sort_values('spearman_r_with_dose', key=abs, ascending=False)
        corr_path = dc_dir / "dose_feature_correlations.csv"
        dose_corr_df.to_csv(corr_path, index=False)
        output_files['dose_feature_correlations'] = corr_path

        # --- 2.  Filter data to EC50 and DMSO for each group ---
        ec50_frames: List[pd.DataFrame] = []
        for (geno, drug), ec50_val in self.ec50_lookup.items():
            grp_mask = (df['genotype'] == geno) & (df['drug'] == drug)
            grp = df.loc[grp_mask]
            if grp.empty:
                continue

            # DMSO controls
            if 'is_control' in grp.columns:
                controls = grp[grp['is_control'] == True]
            else:
                controls = grp[grp['dilut_um'] == 0]

            # Closest dose to EC50
            doses = grp['dilut_um'].dropna().unique()
            non_zero = doses[doses > 0]
            if len(non_zero) > 0:
                closest = non_zero[np.argmin(np.abs(np.log10(non_zero + 1e-12) - np.log10(ec50_val + 1e-12)))]
                ec50_data = grp[np.isclose(grp['dilut_um'], closest, rtol=0.15)]
            else:
                ec50_data = pd.DataFrame()

            if not controls.empty:
                ec50_frames.append(controls)
            if not ec50_data.empty:
                ec50_frames.append(ec50_data)

        # --- 3.  PCA on dose-correlated features at EC50/DMSO ---
        if ec50_frames:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler

            ec50_combined = pd.concat(ec50_frames, ignore_index=True)

            groupby_cols = ['plate', 'well']
            meta_cols = ['genotype', 'drug', 'dilut_string', 'dilut_um',
                         'is_control', 'moa', 'ec50_um']
            groupby_cols = [c for c in groupby_cols if c in ec50_combined.columns]
            meta_cols = [c for c in meta_cols if c in ec50_combined.columns]
            available_feats = [f for f in top_features if f in ec50_combined.columns]

            if len(available_feats) >= 2 and groupby_cols:
                agg_dict = {col: 'median' for col in available_feats}
                for col in meta_cols:
                    agg_dict[col] = 'first'

                profiles = ec50_combined.groupby(groupby_cols).agg(agg_dict).reset_index()
                cc = ec50_combined.groupby(groupby_cols).size().rename('cell_count')
                profiles = profiles.merge(cc.reset_index(), on=groupby_cols)

                X = profiles[available_feats].fillna(0).values
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                n_comp = min(5, len(available_feats), max(len(profiles) - 1, 2))
                if n_comp < 2:
                    n_comp = 2
                pca = PCA(n_components=n_comp, random_state=42)
                pcs = pca.fit_transform(X_scaled)
                for i in range(n_comp):
                    profiles[f'PC{i+1}'] = pcs[:, i]

                pca_path = dc_dir / "dose_correlated_pca_profiles.csv"
                profiles.to_csv(pca_path, index=False)
                output_files['dose_correlated_pca_profiles'] = pca_path

                loadings = pd.DataFrame(
                    pca.components_.T,
                    columns=[f'PC{i+1}' for i in range(n_comp)],
                    index=available_feats,
                )
                load_path = dc_dir / "dose_correlated_pca_loadings.csv"
                loadings.to_csv(load_path)
                output_files['dose_correlated_pca_loadings'] = load_path

                var_exp = pd.DataFrame({
                    'PC': [f'PC{i+1}' for i in range(n_comp)],
                    'variance_explained': pca.explained_variance_,
                    'variance_ratio': pca.explained_variance_ratio_,
                    'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_),
                })
                var_path = dc_dir / "dose_correlated_pca_variance.csv"
                var_exp.to_csv(var_path, index=False)
                output_files['dose_correlated_pca_variance'] = var_path

                self._log_step(
                    step,
                    f"PCA at EC50: {len(profiles)} wells, {len(available_feats)} features, "
                    f"PC1={pca.explained_variance_ratio_[0]*100:.1f}%",
                )

        # --- 4.  Summary by dose for line plots ---
        summary = self.response_summarizer.summarize_by_dose(
            df, top_features, groupby=['genotype', 'drug']
        )
        if not summary.empty:
            summary_path = dc_dir / "dose_correlated_summary_by_dose.csv"
            summary.to_csv(summary_path, index=False)
            output_files['dose_correlated_summary'] = summary_path

        self._log_timing(step, time.time() - start)
        self.checkpoint.mark_complete(step)
        return output_files

    # =================================================================
    # Step 11 – Statistical comparisons
    # =================================================================
    def _step_statistical_comparisons(self) -> Dict[str, Path]:
        step = "11_statistics"
        output_files: Dict[str, Path] = {}

        if self.resume and self.checkpoint.is_complete(step):
            self._log_step(step, "Skipping (already complete)")
            return {}

        self._log_step(step, "Performing statistical comparisons...")
        start = time.time()

        stats_dir = self.output_dir / "statistics"

        fits_path = self.output_dir / "dose_response" / "dose_response_fits.csv"
        if fits_path.exists():
            fits_df = pd.read_csv(fits_path)
            ec50_comparisons = self.statistical_comparator.compare_ec50s(
                fits_df, groupby='genotype', drug_col='drug'
            )
            if not ec50_comparisons.empty:
                ec50_path = stats_dir / "ec50_comparisons.csv"
                ec50_comparisons.to_csv(ec50_path, index=False)
                output_files['ec50_comparisons'] = ec50_path
                self._log_step(step, f"Saved EC50 comparisons: {ec50_path}")

        response_cols = ['foci_count', 'ki67_positive', 'area']
        response_cols = [c for c in response_cols if c in self.raw_data.columns]

        for col in response_cols:
            try:
                comparisons = self.statistical_comparator.compare_responses_at_dose(
                    self.raw_data, col,
                    dose_col='dilut_um', groupby='genotype', drug_col='drug',
                )
                if not comparisons.empty:
                    comp_path = stats_dir / f"genotype_comparison_{col}.csv"
                    comparisons.to_csv(comp_path, index=False)
                    output_files[f'genotype_comparison_{col}'] = comp_path
            except Exception as e:
                self._log_step(step, f"Warning: Could not compare {col}: {e}")

        self._log_timing(step, time.time() - start)
        self.checkpoint.mark_complete(step)
        return output_files

    # =================================================================
    # Step 12 – Generate all plots  (NEW)
    # =================================================================
    def _step_generate_plots(self) -> Dict[str, Path]:
        """Generate all publication-quality plots."""
        step = "12_generate_plots"
        output_files: Dict[str, Path] = {}

        if not HAS_MATPLOTLIB:
            self._log_step(step, "matplotlib not available; skipping plot generation")
            return output_files

        if self.resume and self.checkpoint.is_complete(step):
            self._log_step(step, "Skipping (already complete)")
            return {}

        self._log_step(step, "Generating plots...")
        start = time.time()

        plots_dir = self.output_dir / "plots"

        # ---- A. PCA plots (for each mode × correction) ----
        for mode in ("per_genotype", "across_all"):
            for correction in ("uncorrected", "corrected"):
                label = f"{mode}_{correction}"
                pca_csv_dir = self.output_dir / "pca" / mode / correction
                plot_out = plots_dir / "pca" / mode / correction
                plot_out.mkdir(parents=True, exist_ok=True)

                profiles_path = pca_csv_dir / f"profiles_pca_{label}.csv"
                loadings_path = pca_csv_dir / f"pca_loadings_{label}.csv"
                var_path = pca_csv_dir / f"pca_variance_{label}.csv"

                if not profiles_path.exists():
                    continue

                profiles = pd.read_csv(profiles_path)
                loadings = pd.read_csv(loadings_path, index_col=0)
                var_df = pd.read_csv(var_path)

                var_ratio = dict(zip(var_df['PC'], var_df['variance_ratio']))

                # Scatter PC1 vs PC2
                if 'genotype' in profiles.columns:
                    scatter_path = plot_out / f"pca_scatter_{label}.png"
                    plot_pca_scatter(
                        profiles, 'PC1', 'PC2',
                        color_col='genotype',
                        var_ratio=var_ratio,
                        title=f'PCA — {mode} ({correction})',
                        output_path=scatter_path,
                    )
                    output_files[f"plot_pca_scatter_{label}"] = scatter_path

                # Feature contributions per PC
                plot_pca_feature_contributions(
                    loadings, var_df,
                    n_features=15, n_pcs=5,
                    output_dir=plot_out,
                    label=label,
                )

                # Variance explained
                var_plot = plot_out / f"pca_variance_{label}.png"
                plot_pca_variance_explained(
                    var_df,
                    title=f'Variance Explained — {mode} ({correction})',
                    output_path=var_plot,
                )
                output_files[f"plot_pca_variance_{label}"] = var_plot

                self._log_step(step, f"  PCA plots: {label}")

        # ---- B. Crowding correction plot ----
        if self.crowding_correlations is not None and len(self.crowding_correlations) > 0:
            crowd_plot = plots_dir / "crowding" / "crowding_feature_correlations.png"
            plot_crowding_excluded_features(
                self.crowding_correlations,
                self.crowding_threshold,
                self.crowding_excluded_features,
                output_path=crowd_plot,
            )
            output_files['plot_crowding_correction'] = crowd_plot
            self._log_step(step, "  Crowding exclusion plot saved")

        # ---- C. Dose-correlated feature plots ----
        dc_dir = self.output_dir / "dose_correlated"
        dc_plot_dir = plots_dir / "dose_correlated"

        corr_path = dc_dir / "dose_feature_correlations.csv"
        if corr_path.exists():
            corr_df = pd.read_csv(corr_path)
            dose_corr = pd.Series(
                corr_df['spearman_r_with_dose'].values,
                index=corr_df['feature'].values,
            )
            # Ranking plot
            rank_plot = dc_plot_dir / "dose_correlation_ranking.png"
            plot_dose_correlation_ranking(
                dose_corr, n_top=min(20, len(dose_corr)),
                title='Feature–Dose Correlation Ranking',
                output_path=rank_plot,
            )
            output_files['plot_dose_corr_ranking'] = rank_plot

        # PCA scatter at EC50
        pca_prof_path = dc_dir / "dose_correlated_pca_profiles.csv"
        pca_var_path = dc_dir / "dose_correlated_pca_variance.csv"
        if pca_prof_path.exists() and pca_var_path.exists():
            profiles = pd.read_csv(pca_prof_path)
            var_df = pd.read_csv(pca_var_path)
            var_ratio = dict(zip(var_df['PC'], var_df['variance_ratio']))

            if 'genotype' in profiles.columns:
                ec50_scatter = dc_plot_dir / "dose_correlated_pca_at_ec50.png"
                plot_pca_scatter(
                    profiles, 'PC1', 'PC2',
                    color_col='genotype',
                    var_ratio=var_ratio,
                    title='PCA of Dose-Correlated Features (EC50 + DMSO)',
                    output_path=ec50_scatter,
                )
                output_files['plot_dose_corr_pca'] = ec50_scatter

            # Feature contribution plots for dose-correlated PCA
            dc_load_path = dc_dir / "dose_correlated_pca_loadings.csv"
            if dc_load_path.exists():
                dc_loadings = pd.read_csv(dc_load_path, index_col=0)
                plot_pca_feature_contributions(
                    dc_loadings, var_df,
                    n_features=15, n_pcs=3,
                    output_dir=dc_plot_dir,
                    label="dose_correlated",
                )

        # Line plots: dose-correlated features vs drug dose
        summary_path = dc_dir / "dose_correlated_summary_by_dose.csv"
        if summary_path.exists() and corr_path.exists():
            summary = pd.read_csv(summary_path)
            corr_df = pd.read_csv(corr_path)
            top_features = corr_df.head(15)['feature'].tolist()

            if 'genotype' in summary.columns:
                line_plot = dc_plot_dir / "dose_correlated_feature_lines.png"
                plot_dose_feature_lines(
                    summary, top_features,
                    dose_col='dilut_um',
                    group_col='genotype',
                    title='Dose-Correlated Features vs Drug Dose',
                    output_path=line_plot,
                )
                output_files['plot_dose_feature_lines'] = line_plot
                self._log_step(step, "  Dose-correlated line plots saved")

        self._log_timing(step, time.time() - start)
        self.checkpoint.mark_complete(step)
        return output_files

    # =================================================================
    # Manifest
    # =================================================================
    def _save_manifest(self, output_files: Dict[str, Any]):
        """Save pipeline manifest with metadata."""
        manifest = {
            "experiment_name": self.config.experiment_name,
            "analysis_date": datetime.now().isoformat(),
            "pipeline_version": "2.0.0",
            "configuration": {
                "config_file": str(self.config_path),
                "genotypes": list(self.config.genotypes.keys()),
                "control_label": self.config.control_label,
                "dilut_column": self.config.dilut_column,
                "n_workers": self.n_workers,
                "crowding_threshold": self.crowding_threshold,
            },
            "ec50_values": {
                f"{g}|{d}": v for (g, d), v in self.ec50_lookup.items()
            },
            "data_summary": {
                "total_cells": len(self.raw_data) if self.raw_data is not None else 0,
                "genotypes": (
                    self.raw_data['genotype'].unique().tolist()
                    if self.raw_data is not None and 'genotype' in self.raw_data.columns
                    else []
                ),
                "drugs": (
                    self.raw_data['drug'].unique().tolist()
                    if self.raw_data is not None and 'drug' in self.raw_data.columns
                    else []
                ),
            },
            "feature_counts": {
                "per_genotype_uncorrected": len(self.feature_cols_per_geno),
                "per_genotype_corrected": len(self.feature_cols_per_geno_corrected),
                "across_all_uncorrected": len(self.feature_cols_across),
                "across_all_corrected": len(self.feature_cols_across_corrected),
                "crowding_excluded": len(self.crowding_excluded_features),
            },
            "crowding_excluded_features": self.crowding_excluded_features,
            "timing": self.timing,
            "output_files": {
                k: str(v) if isinstance(v, Path) else v
                for k, v in output_files.items()
            },
        }

        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        self._log_step("manifest", f"Saved: {manifest_path}")


# =============================================================================
# SLURM SUBMISSION SCRIPT GENERATOR
# =============================================================================

def generate_slurm_script(
    config_path: str,
    output_dir: str,
    job_name: str = "dna_damage_pipeline",
    partition: str = "normal",
    time: str = "4:00:00",
    mem: str = "64G",
    cpus: int = 8,
    email: Optional[str] = None,
) -> str:
    """Generate a SLURM submission script."""

    email_line = f"#SBATCH --mail-user={email}\n#SBATCH --mail-type=END,FAIL" if email else ""

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus}
#SBATCH --output={output_dir}/slurm_%j.out
#SBATCH --error={output_dir}/slurm_%j.err
{email_line}

# Load required modules (adjust for your cluster)
# module load python/3.10
# module load anaconda3

# Activate environment (if using conda/venv)
# source activate dna_damage_env

# Print job info
echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Started at: $(date)"
echo "=========================================="

# Run pipeline
python run_dna_damage_pipeline.py \\
    {config_path} \\
    --output {output_dir} \\
    --workers {cpus} \\
    --resume

# Print completion
echo "=========================================="
echo "Finished at: $(date)"
echo "=========================================="
"""
    return script


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="DNA Damage Panel Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run pipeline locally
    python run_dna_damage_pipeline.py config.json --output ./results

    # Run with specific number of workers
    python run_dna_damage_pipeline.py config.json --workers 16

    # Fresh run (ignore checkpoints)
    python run_dna_damage_pipeline.py config.json --no-resume

    # Custom crowding correction threshold
    python run_dna_damage_pipeline.py config.json --crowding-threshold 0.4

    # Generate SLURM submission script
    python run_dna_damage_pipeline.py config.json --generate-slurm
        """
    )

    parser.add_argument(
        "config",
        help="Path to JSON configuration file"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output directory (overrides config)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--resume/--no-resume",
        dest="resume",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Resume from checkpoints (default: True)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--crowding-threshold",
        type=float,
        default=0.3,
        help="|Spearman r| threshold for crowding correction (default: 0.3)"
    )
    parser.add_argument(
        "--generate-slurm",
        action="store_true",
        help="Generate SLURM submission script and exit"
    )
    parser.add_argument(
        "--slurm-partition",
        default="normal",
        help="SLURM partition (default: normal)"
    )
    parser.add_argument(
        "--slurm-time",
        default="4:00:00",
        help="SLURM time limit (default: 4:00:00)"
    )
    parser.add_argument(
        "--slurm-mem",
        default="64G",
        help="SLURM memory (default: 64G)"
    )
    parser.add_argument(
        "--slurm-email",
        help="Email for SLURM notifications"
    )

    args = parser.parse_args()

    # Validate config exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    # Determine output directory
    output_dir = args.output
    if not output_dir:
        with open(config_path) as f:
            config_data = json.load(f)
        output_dir = config_data.get("metadata", {}).get("output_dir", "./output")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate SLURM script if requested
    if args.generate_slurm:
        slurm_script = generate_slurm_script(
            config_path=str(config_path.absolute()),
            output_dir=str(output_dir.absolute()),
            partition=args.slurm_partition,
            time=args.slurm_time,
            mem=args.slurm_mem,
            cpus=args.workers,
            email=args.slurm_email,
        )

        slurm_path = output_dir / "submit_pipeline.slurm"
        with open(slurm_path, 'w') as f:
            f.write(slurm_script)

        print(f"Generated SLURM script: {slurm_path}")
        print(f"\nSubmit with: sbatch {slurm_path}")
        return

    # Run pipeline
    pipeline = DNADamageProductionPipeline(
        config_path=config_path,
        output_dir=output_dir,
        n_workers=args.workers,
        resume=args.resume,
        log_level=args.log_level,
        crowding_threshold=args.crowding_threshold,
    )

    results = pipeline.run()

    # Print summary
    print("\nGenerated files:")
    for name, path in sorted(results.items()):
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
