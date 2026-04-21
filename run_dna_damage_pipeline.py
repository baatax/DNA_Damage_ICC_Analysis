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


def with_timestamped_suffix(base_dir: Path) -> Path:
    """Return a timestamped directory path based on ``base_dir``.

    Example
    -------
    ``/tmp/results`` -> ``/tmp/results_20260408_123045``
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir.parent / f"{base_dir.name}_{timestamp}"


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
    from dna_damage_plotting import PlotGenerator
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
    """
    
    def __init__(
        self,
        config_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        n_workers: int = 4,
        resume: bool = True,
        log_level: str = "INFO",
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
        """
        self.config_path = Path(config_path)
        self.n_workers = n_workers
        self.resume = resume
        
        # Load configuration
        self.config = ExperimentConfig.from_json(config_path)
        
        # Set output directory
        if output_dir:
            base_output_dir = Path(output_dir)
        elif self.config.output_dir:
            base_output_dir = Path(self.config.output_dir)
        else:
            base_output_dir = Path(f"./output_{self.config.experiment_name}")

        self.output_dir = with_timestamped_suffix(base_output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = setup_logging(self.output_dir, log_level)
        
        # Setup checkpoint manager
        self.checkpoint = CheckpointManager(self.output_dir)
        if not resume:
            self.checkpoint.clear()
        
        # Initialize components
        self.loader = DNADamageDataLoader(self.config)
        self.preprocessor = DNADamagePreprocessor(
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
        self.qc_config = self.config.qc

        # Data storage
        self.raw_data: Optional[pd.DataFrame] = None
        self.well_profiles: Optional[pd.DataFrame] = None
        self.well_profiles_norm: Optional[pd.DataFrame] = None
        self.qc_well_table: Optional[pd.DataFrame] = None
        self.feature_cols: List[str] = []
        self.crowding_excluded: List[str] = []

        # Track timing
        self.timing: Dict[str, float] = {}
    
    def _setup_directories(self):
        """Create output directory structure matching ANALYSIS_OVERVIEW."""
        dirs = [
            self.output_dir,
            self.output_dir / "tables",
            self.output_dir / "crowding",
            self.output_dir / "plots" / "qc",
            self.output_dir / "plots" / "normalization",
        ]
        for variant in ("uncorrected", "crowding_corrected"):
            dirs.extend([
                self.output_dir / variant / "tables",
                self.output_dir / variant / "models",
                self.output_dir / variant / "plots" / "embedding",
                self.output_dir / variant / "plots" / "per_channel",
                self.output_dir / variant / "plots" / "clustering",
                self.output_dir / variant / "plots" / "ec50_focused",
                self.output_dir / variant / "plots" / "dmso_controls",
                self.output_dir / variant / "plots" / "per_drug",
                self.output_dir / variant / "plots" / "dose_response",
            ])
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    def _log_step(self, step: str, message: str):
        """Log a pipeline step."""
        self.logger.info(f"[{step}] {message}")
    
    def _log_timing(self, step: str, duration: float):
        """Log timing information."""
        self.timing[step] = duration
        self.logger.info(f"[{step}] Completed in {duration:.2f} seconds")
    
    def run(self) -> Dict[str, Path]:
        """Run the complete analysis pipeline (ANALYSIS_OVERVIEW flow).

        Flow
        ----
        1. Load & concatenate datasets
        2. Single-cell QC filtering
        3. Aggregate to well-level profiles
        4. Well QC + outlier removal
        5. Normalize well-level features
        6. Crowding correction (identify correlated features)
        7. Branch into ``uncorrected`` / ``crowding_corrected`` variants
        8. Per-variant: PCA, per-channel PCA, distance heatmaps,
           EC50-focused PCA, DMSO PCA, per-drug PCA, dose-response
        """
        start_time = time.time()

        self.logger.info("=" * 70)
        self.logger.info("DNA DAMAGE PANEL ANALYSIS PIPELINE")
        self.logger.info("=" * 70)
        self.logger.info(f"Experiment: {self.config.experiment_name}")
        self.logger.info(f"Config: {self.config_path}")
        self.logger.info(f"Output: {self.output_dir}")
        self.logger.info(f"Workers: {self.n_workers}")
        self.logger.info(f"Resume: {self.resume}")
        self.logger.info("=" * 70)

        self._setup_directories()
        output_files: Dict[str, Path] = {}

        try:
            # Step 1: Load & concatenate datasets
            output_files.update(self._step_load_data())

            # Step 2: Single-cell QC filtering
            output_files.update(self._step_single_cell_qc())

            # Step 3: Aggregate to well-level profiles
            output_files.update(self._step_aggregate_wells())

            # Step 4: Well QC + outlier removal
            output_files.update(self._step_well_qc())

            # Step 5: Normalize well-level features
            output_files.update(self._step_normalize())

            # Step 6: Crowding correction
            output_files.update(self._step_crowding_correction())

            # Step 7-8: Run both variants
            for variant in ("uncorrected", "crowding_corrected"):
                output_files.update(self._run_variant(variant))

            # Generate plots from CSV outputs
            output_files.update(self._step_generate_plots(output_files))

            # Save manifest
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
    
    def _step_load_data(self) -> Dict[str, Path]:
        """Step 1: Load & concatenate all parquet data."""
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

        # Identify feature columns once for the whole pipeline
        exclude = set(self.preprocessor.METADATA_COLS + self.preprocessor.QC_COLS)
        numeric_cols = self.raw_data.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_cols = [c for c in numeric_cols if c not in exclude]

        # Cache raw data for resume
        cache_dir = self.output_dir / "cache"
        cache_dir.mkdir(exist_ok=True)
        self.raw_data.to_parquet(cache_dir / "raw_data.parquet", index=False)

        self._log_timing(step, time.time() - start)
        self._log_step(step, f"Loaded {len(self.raw_data):,} cells, {len(self.feature_cols)} features")
        self._log_step(step, f"Genotypes: {self.raw_data['genotype'].unique().tolist()}")
        self._log_step(step, f"Drugs: {self.raw_data['drug'].unique().tolist()}")

        self.checkpoint.mark_complete(step, {
            "n_cells": len(self.raw_data),
            "genotypes": self.raw_data['genotype'].unique().tolist(),
            "drugs": self.raw_data['drug'].unique().tolist(),
            "feature_cols": self.feature_cols,
        })

        return {}
    
    def _step_single_cell_qc(self) -> Dict[str, Path]:
        """Step 2: Single-cell QC filtering (area percentile, etc.)."""
        step = "2_single_cell_qc"
        output_files: Dict[str, Path] = {}

        if self.resume and self.checkpoint.is_complete(step):
            self._log_step(step, "Loading from checkpoint...")
            cache_path = self.output_dir / "cache" / "raw_data_qc.parquet"
            if cache_path.exists():
                self.raw_data = pd.read_parquet(cache_path)
                return {}

        self._log_step(step, "Running single-cell QC...")
        start = time.time()
        n_before = len(self.raw_data)

        # Area-based percentile filtering by plate
        if 'area' in self.raw_data.columns and 'plate' in self.raw_data.columns:
            def _area_filter(group: pd.DataFrame) -> pd.DataFrame:
                lo = group['area'].quantile(0.01)
                hi = group['area'].quantile(0.99)
                return group[(group['area'] >= lo) & (group['area'] <= hi)]
            self.raw_data = self.raw_data.groupby('plate', group_keys=False).apply(_area_filter)

        n_after = len(self.raw_data)
        self._log_step(step, f"Kept {n_after:,}/{n_before:,} cells after area QC")

        # Write cell QC summary
        summary = pd.DataFrame([{
            'cells_before': n_before, 'cells_after': n_after,
            'cells_removed': n_before - n_after,
        }])
        qc_path = self.output_dir / "tables" / "cell_qc_summary.parquet"
        summary.to_parquet(qc_path, index=False)
        output_files['cell_qc_summary'] = qc_path

        cache_dir = self.output_dir / "cache"
        cache_dir.mkdir(exist_ok=True)
        self.raw_data.to_parquet(cache_dir / "raw_data_qc.parquet", index=False)

        self._log_timing(step, time.time() - start)
        self.checkpoint.mark_complete(step)
        return output_files

    def _step_aggregate_wells(self) -> Dict[str, Path]:
        """Step 3: Aggregate single-cell data to well-level profiles."""
        step = "3_aggregate_wells"
        output_files: Dict[str, Path] = {}

        if self.resume and self.checkpoint.is_complete(step):
            self._log_step(step, "Loading from checkpoint...")
            cache_path = self.output_dir / "cache" / "profiles_raw.parquet"
            if cache_path.exists():
                self.well_profiles = pd.read_parquet(cache_path)
                self.feature_cols = self.checkpoint.get_metadata(step).get("feature_cols", self.feature_cols)
                return {}

        self._log_step(step, "Aggregating to well-level profiles...")
        start = time.time()

        groupby_cols = [c for c in ['plate', 'well'] if c in self.raw_data.columns]
        meta_cols = [c for c in ['genotype', 'drug', 'dilut_string', 'dilut_um',
                                  'is_control', 'moa'] if c in self.raw_data.columns]
        avail_feat = [c for c in self.feature_cols if c in self.raw_data.columns]

        agg_dict = {col: 'median' for col in avail_feat}
        for col in meta_cols:
            agg_dict[col] = 'first'

        profiles = self.raw_data.groupby(groupby_cols).agg(agg_dict).reset_index()
        cell_counts = self.raw_data.groupby(groupby_cols).size().rename('cell_count')
        profiles = profiles.merge(cell_counts.reset_index(), on=groupby_cols)

        # Aggregate H2Ax foci and Ki67 at the well level
        grouped = self.raw_data.groupby(groupby_cols)
        if 'foci_count' in self.raw_data.columns:
            profiles['foci_mean_per_cell'] = grouped['foci_count'].mean().values
            profiles['foci_total'] = grouped['foci_count'].sum().values
            profiles['cells_with_foci_fraction'] = grouped['foci_count'].apply(lambda x: (x > 0).mean()).values
            profiles['high_damage_fraction'] = grouped['foci_count'].apply(lambda x: (x > 5).mean()).values
        if 'ki67_positive' in self.raw_data.columns:
            profiles['ki67_positive_fraction'] = grouped['ki67_positive'].mean().values

        self.well_profiles = profiles
        self.feature_cols = avail_feat

        # Write cell QC by well
        qc_by_well_path = self.output_dir / "tables" / "cell_qc_by_well.parquet"
        cell_counts.reset_index().to_parquet(qc_by_well_path, index=False)
        output_files['cell_qc_by_well'] = qc_by_well_path

        # Write raw profiles
        raw_path = self.output_dir / "tables" / "profiles_raw.parquet"
        profiles.to_parquet(raw_path, index=False)
        output_files['profiles_raw'] = raw_path

        cache_dir = self.output_dir / "cache"
        profiles.to_parquet(cache_dir / "profiles_raw.parquet", index=False)

        self._log_step(step, f"Aggregated {len(profiles)} wells, {len(avail_feat)} features")
        self._log_timing(step, time.time() - start)
        self.checkpoint.mark_complete(step, {"feature_cols": self.feature_cols})
        return output_files

    def _step_well_qc(self) -> Dict[str, Path]:
        """Step 4: Well QC — flag outlier wells and optionally remove them."""
        step = "4_well_qc"
        output_files: Dict[str, Path] = {}

        if self.resume and self.checkpoint.is_complete(step):
            self._log_step(step, "Skipping (already complete)")
            cache_path = self.output_dir / "cache" / "profiles_qc.parquet"
            if cache_path.exists():
                self.well_profiles = pd.read_parquet(cache_path)
            return {}

        self._log_step(step, "Running well QC...")
        start = time.time()

        # Compile well-level QC metrics from raw data
        groupby_cols = [c for c in ['plate', 'well'] if c in self.raw_data.columns]
        qc_metrics = self.qc_compiler.compile_well_qc(self.raw_data, groupby_cols)
        qc_decisions = self.qc_compiler.apply_qc_rules(qc_metrics, self.qc_config)
        self.qc_well_table = qc_decisions

        # Save QC flags
        qc_path = self.output_dir / "tables" / "qc_flags.parquet"
        qc_decisions.to_parquet(qc_path, index=False)
        output_files['qc_flags'] = qc_path

        # Also write CSV for plotting
        qc_csv = self.output_dir / "tables" / "well_qc_metrics.csv"
        qc_decisions.to_csv(qc_csv, index=False)
        output_files['well_qc_metrics'] = qc_csv

        # Remove failed wells from profiles
        n_before = len(self.well_profiles)
        removed = pd.DataFrame()
        if 'qc_pass' in qc_decisions.columns:
            pass_wells = qc_decisions[qc_decisions['qc_pass']][groupby_cols].drop_duplicates()
            self.well_profiles = self.well_profiles.merge(pass_wells, on=groupby_cols, how='inner')

            removed = qc_decisions[~qc_decisions['qc_pass']].copy()
            if not removed.empty:
                rm_path = self.output_dir / "tables" / "qc_removed_wells.parquet"
                removed.to_parquet(rm_path, index=False)
                output_files['qc_removed_wells'] = rm_path

                excluded_csv = self.output_dir / "tables" / "excluded_wells.csv"
                removed.to_csv(excluded_csv, index=False)
                output_files['excluded_wells'] = excluded_csv

        # Always emit human-readable QC exclusion report
        report_md = self.output_dir / "tables" / "qc_exclusion_report.md"
        report_cols = [c for c in ["plate", "well", "genotype", "drug", "qc_fail_reasons", "qc_warning_reasons"] if c in qc_decisions.columns]
        with open(report_md, "w", encoding="utf-8") as f:
            f.write("# QC Exclusion Report\n\n")
            f.write(f"- Total wells evaluated: {len(qc_decisions)}\n")
            if "qc_pass" in qc_decisions.columns:
                n_pass = int(qc_decisions["qc_pass"].sum())
                n_fail = int((~qc_decisions["qc_pass"]).sum())
                f.write(f"- Wells passing QC: {n_pass}\n")
                f.write(f"- Wells excluded by QC: {n_fail}\n\n")
            else:
                f.write("- `qc_pass` column not available in QC table.\n\n")

            if removed.empty:
                f.write("## Excluded wells\n\nNo wells were excluded.\n")
            else:
                f.write("## Excluded wells\n\n")
                f.write("The following wells were excluded and why:\n\n")
                f.write("| " + " | ".join(report_cols) + " |\n")
                f.write("|" + "|".join(["---"] * len(report_cols)) + "|\n")
                for row in removed[report_cols].fillna("").itertuples(index=False):
                    f.write("| " + " | ".join(str(x) for x in row) + " |\n")
        output_files["qc_exclusion_report"] = report_md

        n_after = len(self.well_profiles)
        self._log_step(step, f"Kept {n_after}/{n_before} wells after QC")

        cache_dir = self.output_dir / "cache"
        self.well_profiles.to_parquet(cache_dir / "profiles_qc.parquet", index=False)

        self._log_timing(step, time.time() - start)
        self.checkpoint.mark_complete(step)
        return output_files

    def _step_normalize(self) -> Dict[str, Path]:
        """Step 5: Normalize well-level features (control-based or global)."""
        step = "5_normalize"
        output_files: Dict[str, Path] = {}

        if self.resume and self.checkpoint.is_complete(step):
            self._log_step(step, "Loading from checkpoint...")
            cache_path = self.output_dir / "cache" / "profiles_norm.parquet"
            if cache_path.exists():
                self.well_profiles_norm = pd.read_parquet(cache_path)
                return {}

        self._log_step(step, f"Normalizing ({self.config.normalization})...")
        start = time.time()

        from dna_damage_parquet_pipeline import robust_zscore

        df = self.well_profiles.copy()
        avail_feat = [c for c in self.feature_cols if c in df.columns]

        if self.config.normalization == "control_based" and 'is_control' in df.columns:
            # Normalize relative to control medians per plate/genotype/drug
            baseline_cols = [c for c in ['plate', 'genotype', 'drug'] if c in df.columns]
            control = df[df['is_control']]
            if not control.empty and baseline_cols:
                ctrl_medians = control.groupby(baseline_cols)[avail_feat].median().reset_index()
                ctrl_mads = control.groupby(baseline_cols)[avail_feat].apply(
                    lambda x: np.nanmedian(np.abs(x - np.nanmedian(x, axis=0)), axis=0)
                )
                if isinstance(ctrl_mads, pd.DataFrame):
                    ctrl_mads = ctrl_mads.reset_index()
                else:
                    ctrl_mads = None

                merged = df.merge(
                    ctrl_medians, on=baseline_cols, suffixes=('', '_ctrl_med'), how='left',
                )
                for feat in avail_feat:
                    med_col = f"{feat}_ctrl_med"
                    if med_col in merged.columns:
                        merged[feat] = merged[feat] - merged[med_col]
                        merged.drop(columns=[med_col], inplace=True)
                df = merged
            else:
                # Fallback to global robust z-score
                X = df[avail_feat].values.astype(np.float64)
                df[avail_feat] = robust_zscore(X, axis=0)
        else:
            X = df[avail_feat].values.astype(np.float64)
            df[avail_feat] = robust_zscore(X, axis=0)

        # Replace inf with nan
        for feat in avail_feat:
            df[feat] = df[feat].replace([np.inf, -np.inf], np.nan)

        self.well_profiles_norm = df

        norm_path = self.output_dir / "tables" / "profiles_norm.parquet"
        df.to_parquet(norm_path, index=False)
        output_files['profiles_norm'] = norm_path

        # Normalization diagnostics
        diag = {
            "method": self.config.normalization,
            "n_features": len(avail_feat),
            "n_wells": len(df),
        }
        diag_path = self.output_dir / "tables" / "normalization_diagnostics.json"
        with open(diag_path, 'w') as f:
            json.dump(diag, f, indent=2)
        output_files['normalization_diagnostics'] = diag_path

        cache_dir = self.output_dir / "cache"
        df.to_parquet(cache_dir / "profiles_norm.parquet", index=False)

        self._log_timing(step, time.time() - start)
        self._log_step(step, f"Normalized {len(avail_feat)} features for {len(df)} wells")
        self.checkpoint.mark_complete(step)
        return output_files

    def _step_crowding_correction(self) -> Dict[str, Path]:
        """Step 6: Compute crowding-correlated feature exclusions."""
        step = "6_crowding"
        output_files: Dict[str, Path] = {}

        if self.resume and self.checkpoint.is_complete(step):
            self._log_step(step, "Skipping (already complete)")
            self.crowding_excluded = self.checkpoint.get_metadata(step).get("excluded", [])
            return {}

        self._log_step(step, "Computing crowding exclusions...")
        start = time.time()

        profiles = self.well_profiles_norm if self.well_profiles_norm is not None else self.well_profiles
        avail_feat = [c for c in self.feature_cols if c in profiles.columns]

        corr_df, excluded = self.crowding_analyzer.compute_crowding_exclusions(
            profiles, avail_feat, threshold=self.config.crowding_corr_threshold,
        )
        self.crowding_excluded = excluded

        # Also compile crowding summary by drug/dose
        crowding_summary = self.crowding_analyzer.compile_crowding_by_drug_dose(
            self.raw_data, "all",
        )

        crowding_dir = self.output_dir / "crowding"
        if not corr_df.empty:
            corr_path = crowding_dir / "crowding_feature_correlations.csv"
            corr_df.to_csv(corr_path, index=False)
            output_files['crowding_feature_correlations'] = corr_path

        excl_path = crowding_dir / "crowding_excluded_features.txt"
        excl_path.write_text("\n".join(excluded))
        output_files['crowding_excluded_features'] = excl_path

        if not crowding_summary.empty:
            cs_path = crowding_dir / "crowding_by_drug_dose.csv"
            crowding_summary.to_csv(cs_path, index=False)
            output_files['crowding_summary'] = cs_path

        self._log_step(step, f"Excluded {len(excluded)}/{len(avail_feat)} crowding-correlated features")
        self._log_timing(step, time.time() - start)
        self.checkpoint.mark_complete(step, {"excluded": excluded})
        return output_files
    
    # ------------------------------------------------------------------
    # Variant runner (steps 7-8 per ANALYSIS_OVERVIEW sections 6.1-6.9)
    # ------------------------------------------------------------------

    def _run_variant(self, variant: str) -> Dict[str, Path]:
        """Run all per-variant analyses (PCA, dose-response, etc.)."""
        step = f"7_variant_{variant}"
        output_files: Dict[str, Path] = {}

        if self.resume and self.checkpoint.is_complete(step):
            self._log_step(step, "Skipping (already complete)")
            return {}

        self._log_step(step, f"Running variant: {variant}")
        start = time.time()

        from sklearn.decomposition import PCA as PCAModel
        from sklearn.preprocessing import StandardScaler
        from scipy.spatial.distance import pdist, squareform

        profiles = self.well_profiles_norm if self.well_profiles_norm is not None else self.well_profiles
        if profiles is None or profiles.empty:
            self._log_step(step, "No profiles available — skipping variant")
            self.checkpoint.mark_complete(step)
            return output_files

        profiles = profiles.copy()
        avail_feat = [c for c in self.feature_cols if c in profiles.columns]

        # Apply crowding correction for the corrected variant
        if variant == "crowding_corrected" and self.crowding_excluded:
            avail_feat = [c for c in avail_feat if c not in self.crowding_excluded]
            self._log_step(step, f"Using {len(avail_feat)} features (excluded {len(self.crowding_excluded)} crowding-correlated)")
        else:
            self._log_step(step, f"Using {len(avail_feat)} features (uncorrected)")

        if len(avail_feat) < 2:
            self._log_step(step, "Too few features for analysis — skipping variant")
            self.checkpoint.mark_complete(step)
            return output_files

        var_dir = self.output_dir / variant
        tables_dir = var_dir / "tables"
        models_dir = var_dir / "models"
        seed = self.config.random_seed

        # --- Derived features ------------------------------------------------
        if 'dilut_um' in profiles.columns and 'is_control' in profiles.columns:
            baseline_cols = [c for c in ['plate', 'genotype', 'drug'] if c in profiles.columns]
            profiles = self._compute_control_relative_effects(
                profiles, avail_feat, self.config.control_label, baseline_cols,
            )

            # Phenotype magnitude = L2 norm of delta vector
            delta_cols = [f"{c}_delta" for c in avail_feat if f"{c}_delta" in profiles.columns]
            if delta_cols:
                profiles['phenotype_magnitude'] = np.sqrt(
                    (profiles[delta_cols].fillna(0) ** 2).sum(axis=1)
                )
                profiles['phenotype_angle_to_control'] = profiles['phenotype_magnitude'].apply(
                    lambda x: np.degrees(np.arccos(np.clip(1 / (1 + x), -1, 1)))
                )

        delta_path = tables_dir / "profiles_delta.parquet"
        profiles.to_parquet(delta_path, index=False)
        output_files[f'{variant}/profiles_delta'] = delta_path

        # --- Feature selection -----------------------------------------------
        selected = self.feature_selector.select_features(profiles, avail_feat)
        if len(selected) < 2:
            self._log_step(step, "Too few features after selection — skipping PCA")
            self.checkpoint.mark_complete(step)
            return output_files

        # --- PCA embedding + loadings ----------------------------------------
        X = profiles[selected].fillna(0).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        n_comp = min(self.config.pca_components, len(selected), len(profiles) - 1)
        n_comp = max(n_comp, 2)

        pca = PCAModel(n_components=n_comp, random_state=seed)
        pcs = pca.fit_transform(X_scaled)
        for i in range(n_comp):
            profiles[f'PC{i+1}'] = pcs[:, i]

        # Save PCA profiles CSV (for plotting)
        pca_csv = var_dir / "plots" / "embedding" / f"profiles_pca_{variant}.csv"
        profiles.to_csv(pca_csv, index=False)
        output_files[f'{variant}/profiles_pca'] = pca_csv

        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(n_comp)],
            index=selected,
        )
        loadings_csv = var_dir / "plots" / "embedding" / f"pca_loadings_{variant}.csv"
        loadings.to_csv(loadings_csv)
        output_files[f'{variant}/pca_loadings'] = loadings_csv

        var_exp = pd.DataFrame({
            'PC': [f'PC{i+1}' for i in range(n_comp)],
            'variance_explained': pca.explained_variance_,
            'variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_),
        })
        var_csv = var_dir / "plots" / "embedding" / f"pca_variance_{variant}.csv"
        var_exp.to_csv(var_csv, index=False)
        output_files[f'{variant}/pca_variance'] = var_csv

        pca_info = {
            'n_components': n_comp,
            'n_features': len(selected),
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'variant': variant,
        }
        pca_info_path = models_dir / "pca_info.json"
        with open(pca_info_path, 'w') as f:
            json.dump(pca_info, f, indent=2)
        output_files[f'{variant}/pca_info'] = pca_info_path

        self._log_step(step, f"PCA: {len(selected)} features, "
                       f"PC1={pca.explained_variance_ratio_[0]*100:.1f}%, "
                       f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%")

        # --- Per-channel PCA -------------------------------------------------
        channel_prefixes = set()
        for feat in selected:
            parts = feat.rsplit('_', 1)
            if len(parts) == 2:
                channel_prefixes.add(parts[0])

        for prefix in sorted(channel_prefixes):
            ch_feats = [f for f in selected if f.startswith(prefix + '_') or f == prefix]
            if len(ch_feats) < 2:
                continue
            ch_X = profiles[ch_feats].fillna(0).values
            ch_scaled = StandardScaler().fit_transform(ch_X)
            n_ch = min(3, len(ch_feats), len(profiles) - 1)
            if n_ch < 2:
                continue
            ch_pca = PCAModel(n_components=n_ch, random_state=seed)
            ch_pca.fit(ch_scaled)
            ch_dir = var_dir / "plots" / "per_channel" / prefix
            ch_dir.mkdir(parents=True, exist_ok=True)
            ch_info = {
                'channel': prefix, 'n_features': len(ch_feats),
                'explained_variance_ratio': ch_pca.explained_variance_ratio_.tolist(),
            }
            with open(ch_dir / "pca_info.json", 'w') as f:
                json.dump(ch_info, f, indent=2)

        # --- Global distance heatmap -----------------------------------------
        label_cols = [c for c in ['genotype', 'drug', 'dilut_string'] if c in profiles.columns]
        if label_cols:
            profiles['_sample_label'] = profiles[label_cols].astype(str).agg(' | '.join, axis=1)
            label_means = profiles.groupby('_sample_label')[selected].mean()
            if len(label_means) >= 2:
                dist = squareform(pdist(label_means.values, metric='correlation'))
                dist_df = pd.DataFrame(dist, index=label_means.index, columns=label_means.index)
                dist_csv = var_dir / "plots" / "clustering" / f"distance_heatmap_{variant}.csv"
                dist_df.to_csv(dist_csv)
                output_files[f'{variant}/distance_heatmap'] = dist_csv
            profiles.drop(columns=['_sample_label'], inplace=True)

        # --- EC50-focused PCA ------------------------------------------------
        # Pick the single dose closest to the configured EC50 (in log-space)
        # per (genotype, drug), so each group contributes only its EC50-matched
        # replicate wells rather than every dose inside a ±window.
        ec50_rows = pd.DataFrame()
        if 'dilut_um' in profiles.columns:
            for geno_name, geno_cfg in self.config.genotypes.items():
                for drug_name, drug_cfg in geno_cfg.drugs.items():
                    if drug_cfg.ec50_um is None:
                        continue
                    group = profiles[
                        (profiles.get('genotype') == geno_name)
                        & (profiles.get('drug') == drug_name)
                        & (profiles['dilut_um'] > 0)
                        & profiles['dilut_um'].notna()
                    ]
                    if group.empty:
                        continue
                    available_doses = group['dilut_um'].unique()
                    closest_dose = min(
                        available_doses,
                        key=lambda d: abs(np.log10(d) - np.log10(drug_cfg.ec50_um)),
                    )
                    ec50_rows = pd.concat(
                        [ec50_rows, group[group['dilut_um'] == closest_dose]]
                    )

        if len(ec50_rows) >= 3 and len(selected) >= 2:
            ec50_dir = var_dir / "plots" / "ec50_focused"
            ec50_csv = ec50_dir / f"ec50_profiles_{variant}.csv"
            ec50_rows.to_csv(ec50_csv, index=False)
            output_files[f'{variant}/ec50_profiles'] = ec50_csv

        # --- DMSO/control-only PCA -------------------------------------------
        if 'is_control' in profiles.columns:
            ctrl = profiles[profiles['is_control']]
            if len(ctrl) >= 3:
                ctrl_dir = var_dir / "plots" / "dmso_controls"
                ctrl_csv = ctrl_dir / f"dmso_profiles_{variant}.csv"
                ctrl.to_csv(ctrl_csv, index=False)
                output_files[f'{variant}/dmso_profiles'] = ctrl_csv

                summary_info = {'n_control_wells': len(ctrl)}
                with open(ctrl_dir / "selection_summary.json", 'w') as f:
                    json.dump(summary_info, f, indent=2)

        # --- Per-drug PCA ----------------------------------------------------
        if 'drug' in profiles.columns:
            for drug, drug_df in profiles.groupby('drug'):
                if len(drug_df) < 3:
                    continue
                drug_safe = str(drug).replace("/", "_").replace("\\", "_")
                drug_dir = var_dir / "plots" / "per_drug"
                drug_csv = drug_dir / f"profiles_{drug_safe}_{variant}.csv"
                drug_df.to_csv(drug_csv, index=False)
                output_files[f'{variant}/per_drug_{drug_safe}'] = drug_csv

        # --- Dose-response modeling ------------------------------------------
        output_files.update(self._variant_dose_response(profiles, avail_feat, variant))

        self._log_timing(step, time.time() - start)
        self.checkpoint.mark_complete(step)
        return output_files

    def _variant_dose_response(
        self,
        profiles: pd.DataFrame,
        feature_cols: List[str],
        variant: str,
    ) -> Dict[str, Path]:
        """Dose-response fitting within a variant (H2Ax foci + Ki67 included)."""
        output_files: Dict[str, Path] = {}
        var_dir = self.output_dir / variant
        tables_dir = var_dir / "tables"
        models_dir = var_dir / "models"
        dr_plot_dir = var_dir / "plots" / "dose_response"

        # Build response columns — include H2Ax foci and Ki67 metrics
        response_candidates = [
            'foci_count', 'foci_mean_per_cell', 'ki67_positive', 'ki67_positive_fraction',
            'gamma_h2ax_mean_intensity', 'area', 'dapi_mean_intensity',
            'phenotype_magnitude',
        ]
        available = profiles.columns.tolist()
        response_cols = [c for c in response_candidates if c in available]

        # Also include delta versions
        effect_cols = []
        for col in response_cols:
            delta = f"{col}_delta"
            effect_cols.append(delta if delta in available else col)
        effect_cols = list(dict.fromkeys(effect_cols))  # dedupe

        if not effect_cols:
            return output_files

        # Fit dose-response curves
        all_fits = []
        for col in effect_cols:
            try:
                fits = self.dose_response_analyzer.fit_drug_response(
                    profiles, col, groupby=['genotype', 'drug'],
                    weight_column='cell_count' if 'cell_count' in profiles.columns else None,
                )
                if not fits.empty:
                    all_fits.append(fits)
            except Exception as e:
                self._log_step(f"dose_response_{variant}", f"Warning: {col}: {e}")

        if all_fits:
            fits_df = pd.concat(all_fits, ignore_index=True)
            fits_csv = dr_plot_dir / "dose_response_fits.csv"
            fits_df.to_csv(fits_csv, index=False)
            output_files[f'{variant}/dose_response_fits'] = fits_csv

            fits_json = models_dir / "dose_response_fits.json"
            fits_df.to_json(fits_json, orient='records', indent=2)
            output_files[f'{variant}/dose_response_fits_json'] = fits_json

        # Dose-response metrics per feature
        per_feat_fits = []
        for feat in feature_cols[:20]:  # cap to avoid very long runs
            delta = f"{feat}_delta"
            col = delta if delta in available else feat
            if col not in available:
                continue
            try:
                fits = self.dose_response_analyzer.fit_drug_response(
                    profiles, col, groupby=['genotype', 'drug'],
                )
                if not fits.empty:
                    per_feat_fits.append(fits)
            except Exception:
                continue
        if per_feat_fits:
            pf_df = pd.concat(per_feat_fits, ignore_index=True)
            pf_path = tables_dir / "per_feature_dose_response_metrics.parquet"
            pf_df.to_parquet(pf_path, index=False)
            output_files[f'{variant}/per_feature_dr'] = pf_path

        # Summary statistics by dose
        summary = self.response_summarizer.summarize_by_dose(
            profiles, effect_cols, groupby=['genotype', 'drug'],
        )
        if not summary.empty:
            sm_path = dr_plot_dir / "dose_response_summary.csv"
            summary.to_csv(sm_path, index=False)
            output_files[f'{variant}/dose_response_summary'] = sm_path

        # Fold-change vs control
        fc = self.response_summarizer.compute_fold_change_vs_control(
            profiles, effect_cols, groupby=['genotype', 'drug'], baseline_cols=['plate'],
        )
        if not fc.empty:
            fc_path = dr_plot_dir / "fold_change_vs_control.csv"
            fc.to_csv(fc_path, index=False)
            output_files[f'{variant}/fold_change'] = fc_path

        # Bootstrap EC50 comparisons
        for col in effect_cols[:5]:
            try:
                ec50_boot = self.statistical_comparator.compare_ec50s_bootstrap(
                    profiles, col, dose_col='dilut_um', groupby='genotype', drug_col='drug',
                    weight_col='cell_count' if 'cell_count' in profiles.columns else None,
                    n_boot=200,
                )
                if not ec50_boot.empty:
                    bp = tables_dir / f"ec50_bootstrap_{col}.csv"
                    ec50_boot.to_csv(bp, index=False)
                    output_files[f'{variant}/ec50_bootstrap_{col}'] = bp
            except Exception:
                continue

        # Genotype response comparisons
        for col in effect_cols[:5]:
            try:
                comp = self.statistical_comparator.compare_responses_at_dose(
                    profiles, col, dose_col='dilut_um', groupby='genotype', drug_col='drug',
                )
                if not comp.empty:
                    cp = tables_dir / f"genotype_comparison_{col}.csv"
                    comp.to_csv(cp, index=False)
                    output_files[f'{variant}/genotype_comparison_{col}'] = cp
            except Exception:
                continue

        return output_files

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _step_generate_plots(self, output_files: Dict[str, Path]) -> Dict[str, Path]:
        """Generate plots from CSV outputs."""
        step = "8_generate_plots"
        plot_files: Dict[str, Path] = {}

        existing_plot_count = len(list((self.output_dir / "plots").rglob("*.png")))
        if self.resume and self.checkpoint.is_complete(step) and existing_plot_count > 0:
            self._log_step(step, "Skipping (already complete)")
            return plot_files

        self._log_step(step, "Generating plots for CSV outputs...")
        start = time.time()

        csv_paths = [Path(p) for p in output_files.values() if str(p).endswith(".csv")]
        if not csv_paths:
            csv_paths = list(self.output_dir.rglob("*.csv"))
        if not csv_paths:
            self._log_step(step, "No CSV outputs available for plotting")
            self.checkpoint.mark_complete(step)
            return plot_files

        plotter = PlotGenerator(self.output_dir, logger=self.logger)
        results = plotter.generate_plots(csv_paths)
        for result in results:
            for plot_path in result.plot_paths:
                key = f"plot::{result.csv_path.stem}::{plot_path.name}"
                plot_files[key] = plot_path

        self._log_timing(step, time.time() - start)
        self._log_step(step, f"Generated {len(plot_files)} plots")
        self.checkpoint.mark_complete(step)

        return plot_files

    @staticmethod
    def _compute_control_relative_effects(
        well_df: pd.DataFrame,
        feature_cols: List[str],
        control_label: str,
        baseline_cols: List[str],
    ) -> pd.DataFrame:
        """Compute control-relative effects per well."""
        def mad(series: pd.Series) -> float:
            return float(np.nanmedian(np.abs(series - np.nanmedian(series))))

        df = well_df.copy()
        if not baseline_cols:
            df['__baseline__'] = "all"
            baseline_cols = ['__baseline__']
        if 'is_control' not in df.columns:
            if 'dilut_string' in df.columns:
                df['is_control'] = df['dilut_string'].str.upper() == control_label.upper()
            elif 'dilut_um' in df.columns:
                df['is_control'] = df['dilut_um'] == 0

        control_df = df[df['is_control']].copy()
        if control_df.empty:
            return df

        stats = control_df.groupby(baseline_cols)[feature_cols].agg(['median', mad])
        stats.columns = [
            f"{col}_{stat_name}"
            for col, stat_name in stats.columns
        ]
        stats = stats.reset_index()

        df = df.merge(stats, on=baseline_cols, how='left')

        for col in feature_cols:
            median_col = f"{col}_median"
            mad_col = f"{col}_mad"
            if median_col not in df.columns:
                continue
            df[f"{col}_delta"] = df[col] - df[median_col]
            df[f"{col}_log2fc"] = np.where(
                (df[col] > 0) & (df[median_col] > 0),
                np.log2(df[col] / df[median_col]),
                np.nan,
            )
            if mad_col in df.columns:
                df[f"{col}_robust_z"] = np.where(
                    df[mad_col] > 0,
                    (df[col] - df[median_col]) / df[mad_col],
                    np.nan,
                )

        return df
    
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
                "normalization": self.config.normalization,
                "crowding_corr_threshold": self.config.crowding_corr_threshold,
                "pca_components": self.config.pca_components,
                "random_seed": self.config.random_seed,
                "n_workers": self.n_workers,
                "qc": self.qc_config.__dict__ if self.qc_config else None,
            },
            "data_summary": {
                "total_cells": len(self.raw_data) if self.raw_data is not None else 0,
                "n_wells": len(self.well_profiles) if self.well_profiles is not None else 0,
                "n_features": len(self.feature_cols),
                "n_crowding_excluded": len(self.crowding_excluded),
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
            "variants": ["uncorrected", "crowding_corrected"],
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
    )
    
    results = pipeline.run()
    
    # Print summary
    print("\nGenerated files:")
    for name, path in sorted(results.items()):
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
