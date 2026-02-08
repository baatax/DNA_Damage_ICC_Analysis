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
        self.qc_config = self.config.qc
        
        # Data storage
        self.raw_data: Optional[pd.DataFrame] = None
        self.data_per_geno: Optional[pd.DataFrame] = None
        self.data_across_all: Optional[pd.DataFrame] = None
        self.filtered_raw_data: Optional[pd.DataFrame] = None
        self.qc_well_table: Optional[pd.DataFrame] = None
        self.well_profiles: Dict[str, pd.DataFrame] = {}
        self.well_effects: Dict[str, pd.DataFrame] = {}
        self.feature_cols_per_geno: List[str] = []
        self.feature_cols_across: List[str] = []
        
        # Track timing
        self.timing: Dict[str, float] = {}
    
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
            self.output_dir / "pca" / "per_genotype",
            self.output_dir / "pca" / "across_all",
            self.output_dir / "dose_response",
            self.output_dir / "statistics",
        ]
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
        """
        Run the complete analysis pipeline.
        
        Returns
        -------
        dict
            Dictionary mapping output names to file paths
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
        output_files = {}
        
        try:
            # Step 1: Load data
            output_files.update(self._step_load_data())
            
            # Step 2: Preprocess per-genotype
            output_files.update(self._step_preprocess_per_genotype())
            
            # Step 3: Preprocess across-all
            output_files.update(self._step_preprocess_across_all())
            
            # Step 4: Compile crowding metrics
            output_files.update(self._step_compile_crowding())
            
            # Step 5: Compile QC metrics
            output_files.update(self._step_compile_qc())
            
            # Step 6: Generate per-well CSVs
            output_files.update(self._step_generate_per_well())
            
            # Step 7: Compute profiles and PCA
            output_files.update(self._step_compute_pca())
            
            # Step 8: Dose-response analysis
            output_files.update(self._step_dose_response_analysis())
            
            # Step 9: Statistical comparisons
            output_files.update(self._step_statistical_comparisons())
            
            # Step 10: Generate plots from CSV outputs
            output_files.update(self._step_generate_plots(output_files))
            
            # Step 11: Save manifest
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
    
    def _step_preprocess_per_genotype(self) -> Dict[str, Path]:
        """Step 2: Preprocess with per-genotype normalization."""
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
        
        # Cache
        cache_dir = self.output_dir / "cache"
        self.data_per_geno.to_parquet(cache_dir / "data_per_geno.parquet", index=False)
        
        self._log_timing(step, time.time() - start)
        self._log_step(step, f"Feature columns: {len(self.feature_cols_per_geno)}")
        
        self.checkpoint.mark_complete(step, {
            "feature_cols": self.feature_cols_per_geno,
        })
        
        return {}
    
    def _step_preprocess_across_all(self) -> Dict[str, Path]:
        """Step 3: Preprocess with across-all normalization."""
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
        
        # Cache
        cache_dir = self.output_dir / "cache"
        self.data_across_all.to_parquet(cache_dir / "data_across_all.parquet", index=False)
        
        self._log_timing(step, time.time() - start)
        self._log_step(step, f"Feature columns: {len(self.feature_cols_across)}")
        
        self.checkpoint.mark_complete(step, {
            "feature_cols": self.feature_cols_across,
        })
        
        return {}
    
    def _step_compile_crowding(self) -> Dict[str, Path]:
        """Step 4: Compile crowding metrics per drug/dose."""
        step = "4_compile_crowding"
        output_files = {}
        
        if self.resume and self.checkpoint.is_complete(step):
            self._log_step(step, "Skipping (already complete)")
            return {}
        
        self._log_step(step, "Compiling crowding metrics...")
        start = time.time()
        
        # Per-genotype crowding
        crowding_per_geno = self.crowding_analyzer.compile_crowding_by_drug_dose(
            self.data_per_geno, "per_genotype"
        )
        
        # Across-all crowding
        crowding_across = self.crowding_analyzer.compile_crowding_by_drug_dose(
            self.data_across_all, "across_all"
        )
        
        # Combine
        crowding_combined = pd.concat(
            [crowding_per_geno, crowding_across],
            ignore_index=True
        )
        
        # Save
        crowding_path = self.output_dir / "crowding" / "crowding_by_drug_dose.csv"
        crowding_combined.to_csv(crowding_path, index=False)
        output_files['crowding'] = crowding_path
        
        # Also save separate files
        if not crowding_per_geno.empty:
            path = self.output_dir / "crowding" / "crowding_per_genotype.csv"
            crowding_per_geno.to_csv(path, index=False)
            output_files['crowding_per_genotype'] = path
        
        if not crowding_across.empty:
            path = self.output_dir / "crowding" / "crowding_across_all.csv"
            crowding_across.to_csv(path, index=False)
            output_files['crowding_across_all'] = path
        
        self._log_timing(step, time.time() - start)
        self._log_step(step, f"Saved: {crowding_path}")
        
        self.checkpoint.mark_complete(step)
        
        return output_files
    
    def _step_compile_qc(self) -> Dict[str, Path]:
        """Step 5: Compile QC metrics per well."""
        step = "5_compile_qc"
        output_files = {}
        
        if self.resume and self.checkpoint.is_complete(step):
            self._log_step(step, "Skipping (already complete)")
            return {}
        
        self._log_step(step, "Compiling QC metrics...")
        start = time.time()
        
        groupby_cols = ['plate', 'well']
        groupby_cols = [c for c in groupby_cols if c in self.raw_data.columns]
        
        qc_metrics = self.qc_compiler.compile_well_qc(self.raw_data, groupby_cols)
        qc_decisions = self.qc_compiler.apply_qc_rules(qc_metrics, self.qc_config)
        self.qc_well_table = qc_decisions
        self.filtered_raw_data = self._filter_by_qc(self.raw_data)

        # Save
        qc_path = self.output_dir / "qc" / "well_qc_metrics.csv"
        qc_decisions.to_csv(qc_path, index=False)
        output_files['qc'] = qc_path

        if self.qc_well_table is not None:
            excluded = self.qc_well_table[~self.qc_well_table['qc_pass']]
            excluded_path = self.output_dir / "qc" / "excluded_wells.csv"
            excluded.to_csv(excluded_path, index=False)
            output_files['excluded_wells'] = excluded_path
        
        self._log_timing(step, time.time() - start)
        self._log_step(step, f"Saved: {qc_path}")
        
        self.checkpoint.mark_complete(step)
        
        return output_files
    
    def _step_generate_per_well(self) -> Dict[str, Path]:
        """Step 6: Generate per-well aggregated CSVs."""
        step = "6_generate_per_well"
        output_files = {}
        
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

            df = self._filter_by_qc(df)

            # Group by well and aggregate
            groupby_cols = ['plate', 'well']
            meta_cols = ['genotype', 'drug', 'dilut_string', 'dilut_um', 'is_control', 'moa']
            
            groupby_cols = [c for c in groupby_cols if c in df.columns]
            meta_cols = [c for c in meta_cols if c in df.columns]
            
            if not groupby_cols:
                continue
            
            # Aggregate features
            agg_dict = {col: 'median' for col in feature_cols if col in df.columns}
            for col in meta_cols:
                agg_dict[col] = 'first'
            
            well_data = df.groupby(groupby_cols).agg(agg_dict).reset_index()
            cell_counts = df.groupby(groupby_cols).size().rename('cell_count')
            well_data = well_data.merge(cell_counts.reset_index(), on=groupby_cols)

            baseline_cols = [c for c in ['plate', 'genotype', 'drug'] if c in well_data.columns]
            effects = self._compute_control_relative_effects(
                well_data,
                feature_cols,
                self.config.control_label,
                baseline_cols,
            )

            # Save combined file
            combined_path = mode_dir / f"well_profiles_{mode}.csv"
            well_data.to_csv(combined_path, index=False)
            output_files[f"well_profiles_{mode}"] = combined_path

            effects_path = mode_dir / f"well_effects_{mode}.csv"
            effects.to_csv(effects_path, index=False)
            output_files[f"well_effects_{mode}"] = effects_path
            self.well_profiles[mode] = well_data
            self.well_effects[mode] = effects
            
            # Save per-genotype files
            if 'genotype' in well_data.columns:
                for geno in well_data['genotype'].unique():
                    geno_data = well_data[well_data['genotype'] == geno]
                    geno_safe = geno.replace("/", "_").replace("\\", "_")
                    geno_path = mode_dir / f"well_profiles_{geno_safe}_{mode}.csv"
                    geno_data.to_csv(geno_path, index=False)
                    output_files[f"well_profiles_{geno_safe}_{mode}"] = geno_path

                    geno_effects = effects[effects['genotype'] == geno]
                    geno_effects_path = mode_dir / f"well_effects_{geno_safe}_{mode}.csv"
                    geno_effects.to_csv(geno_effects_path, index=False)
                    output_files[f"well_effects_{geno_safe}_{mode}"] = geno_effects_path
            
            self._log_step(step, f"Saved {mode} well profiles: {len(well_data)} wells")
        
        self._log_timing(step, time.time() - start)
        
        self.checkpoint.mark_complete(step)
        
        return output_files
    
    def _step_compute_pca(self) -> Dict[str, Path]:
        """Step 7: Compute PCA on aggregated profiles."""
        step = "7_compute_pca"
        output_files = {}
        
        if self.resume and self.checkpoint.is_complete(step):
            self._log_step(step, "Skipping (already complete)")
            return {}
        
        self._log_step(step, "Computing PCA...")
        start = time.time()
        
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        for mode, df, feature_cols in [
            ("per_genotype", self.data_per_geno, self.feature_cols_per_geno),
            ("across_all", self.data_across_all, self.feature_cols_across),
        ]:
            pca_dir = self.output_dir / "pca" / mode
            pca_dir.mkdir(parents=True, exist_ok=True)

            df = self._filter_by_qc(df)
            
            # Select features
            selected_features = self.feature_selector.select_features(df, feature_cols)
            
            if len(selected_features) < 2:
                self._log_step(step, f"Warning: Insufficient features for PCA ({mode})")
                continue
            
            # Aggregate to well level
            groupby_cols = ['plate', 'well']
            meta_cols = ['genotype', 'drug', 'dilut_string', 'dilut_um', 'is_control', 'moa']
            
            groupby_cols = [c for c in groupby_cols if c in df.columns]
            meta_cols = [c for c in meta_cols if c in df.columns]
            
            if not groupby_cols:
                continue
            
            agg_dict = {col: 'median' for col in selected_features}
            for col in meta_cols:
                agg_dict[col] = 'first'
            
            profiles = df.groupby(groupby_cols).agg(agg_dict).reset_index()
            cell_counts = df.groupby(groupby_cols).size().rename('cell_count')
            profiles = profiles.merge(cell_counts.reset_index(), on=groupby_cols)
            
            # Compute PCA
            X = profiles[selected_features].fillna(0).values
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            n_components = min(10, len(selected_features), len(profiles) - 1)
            if n_components < 2:
                n_components = 2
            
            pca = PCA(n_components=n_components, random_state=42)
            pcs = pca.fit_transform(X_scaled)
            
            for i in range(n_components):
                profiles[f'PC{i+1}'] = pcs[:, i]
            
            # Save profiles with PCA
            profiles_path = pca_dir / f"profiles_pca_{mode}.csv"
            profiles.to_csv(profiles_path, index=False)
            output_files[f"profiles_pca_{mode}"] = profiles_path
            
            # Save PCA loadings
            loadings = pd.DataFrame(
                pca.components_.T,
                columns=[f'PC{i+1}' for i in range(n_components)],
                index=selected_features
            )
            loadings_path = pca_dir / f"pca_loadings_{mode}.csv"
            loadings.to_csv(loadings_path)
            output_files[f"pca_loadings_{mode}"] = loadings_path
            
            # Save variance explained
            var_explained = pd.DataFrame({
                'PC': [f'PC{i+1}' for i in range(n_components)],
                'variance_explained': pca.explained_variance_,
                'variance_ratio': pca.explained_variance_ratio_,
                'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_),
            })
            var_path = pca_dir / f"pca_variance_{mode}.csv"
            var_explained.to_csv(var_path, index=False)
            output_files[f"pca_variance_{mode}"] = var_path
            
            self._log_step(step, f"{mode}: {len(selected_features)} features, "
                          f"PC1={pca.explained_variance_ratio_[0]*100:.1f}%, "
                          f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%")
        
        self._log_timing(step, time.time() - start)
        
        self.checkpoint.mark_complete(step)
        
        return output_files
    
    def _step_dose_response_analysis(self) -> Dict[str, Path]:
        """Step 8: Perform dose-response analysis."""
        step = "8_dose_response"
        output_files = {}
        
        if self.resume and self.checkpoint.is_complete(step):
            self._log_step(step, "Skipping (already complete)")
            return {}
        
        self._log_step(step, "Performing dose-response analysis...")
        start = time.time()
        
        dr_dir = self.output_dir / "dose_response"
        
        effects_df = self.well_effects.get("per_genotype")
        if effects_df is None or effects_df.empty:
            effects_path = self.output_dir / "per_well" / "per_genotype" / "well_effects_per_genotype.csv"
            if effects_path.exists():
                effects_df = pd.read_csv(effects_path)

        analysis_df = effects_df if effects_df is not None and not effects_df.empty else self.filtered_raw_data
        if analysis_df is None or analysis_df.empty:
            analysis_df = self.raw_data

        response_cols = [
            'foci_count', 'foci_mean_per_cell', 'ki67_positive',
            'gamma_h2ax_mean_intensity', 'area', 'dapi_mean_intensity'
        ]
        available_cols = analysis_df.columns.tolist()
        response_cols = [c for c in response_cols if c in available_cols]
        effect_response_cols = []
        for col in response_cols:
            delta_col = f"{col}_delta"
            effect_response_cols.append(delta_col if delta_col in available_cols else col)
        
        if not response_cols:
            self._log_step(step, "Warning: No response columns found for dose-response analysis")
            self.checkpoint.mark_complete(step)
            return output_files
        
        # Fit dose-response curves per genotype/drug
        all_fits = []
        for col in effect_response_cols:
            try:
                fits = self.dose_response_analyzer.fit_drug_response(
                    analysis_df,
                    col,
                    groupby=['genotype', 'drug'],
                    weight_column='cell_count' if 'cell_count' in analysis_df.columns else None,
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
            self._log_step(step, f"Saved dose-response fits: {fits_path}")
        
        # Summary statistics by dose
        summary = self.response_summarizer.summarize_by_dose(
            analysis_df,
            effect_response_cols,
            groupby=['genotype', 'drug']
        )
        if not summary.empty:
            summary_path = dr_dir / "dose_response_summary.csv"
            summary.to_csv(summary_path, index=False)
            output_files['dose_response_summary'] = summary_path
        
        # Fold-change vs control
        fold_change = self.response_summarizer.compute_fold_change_vs_control(
            analysis_df,
            effect_response_cols,
            groupby=['genotype', 'drug'],
            baseline_cols=['plate']
        )
        if not fold_change.empty:
            fc_path = dr_dir / "fold_change_vs_control.csv"
            fold_change.to_csv(fc_path, index=False)
            output_files['fold_change'] = fc_path
        
        self._log_timing(step, time.time() - start)
        
        self.checkpoint.mark_complete(step)
        
        return output_files
    
    def _step_statistical_comparisons(self) -> Dict[str, Path]:
        """Step 9: Perform statistical comparisons between genotypes."""
        step = "9_statistics"
        output_files = {}
        
        if self.resume and self.checkpoint.is_complete(step):
            self._log_step(step, "Skipping (already complete)")
            return {}
        
        self._log_step(step, "Performing statistical comparisons...")
        start = time.time()
        
        stats_dir = self.output_dir / "statistics"
        
        effects_df = self.well_effects.get("per_genotype")
        if effects_df is None or effects_df.empty:
            effects_path = self.output_dir / "per_well" / "per_genotype" / "well_effects_per_genotype.csv"
            if effects_path.exists():
                effects_df = pd.read_csv(effects_path)

        analysis_df = effects_df if effects_df is not None and not effects_df.empty else self.filtered_raw_data
        if analysis_df is None or analysis_df.empty:
            analysis_df = self.raw_data

        response_cols = ['foci_count', 'ki67_positive', 'area']
        available_cols = analysis_df.columns.tolist()
        response_cols = [c for c in response_cols if c in available_cols]
        effect_response_cols = []
        for col in response_cols:
            delta_col = f"{col}_delta"
            effect_response_cols.append(delta_col if delta_col in available_cols else col)

        for col in effect_response_cols:
            ec50_boot = self.statistical_comparator.compare_ec50s_bootstrap(
                analysis_df,
                col,
                dose_col='dilut_um',
                groupby='genotype',
                drug_col='drug',
                weight_col='cell_count' if 'cell_count' in analysis_df.columns else None,
            )

            if not ec50_boot.empty:
                ec50_path = stats_dir / f"ec50_bootstrap_{col}.csv"
                ec50_boot.to_csv(ec50_path, index=False)
                output_files[f'ec50_bootstrap_{col}'] = ec50_path
                self._log_step(step, f"Saved EC50 bootstrap comparisons: {ec50_path}")
        
        # Compare responses at each dose
        for col in effect_response_cols:
            try:
                comparisons = self.statistical_comparator.compare_responses_at_dose(
                    analysis_df,
                    col,
                    dose_col='dilut_um',
                    groupby='genotype',
                    drug_col='drug'
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

    def _step_generate_plots(self, output_files: Dict[str, Path]) -> Dict[str, Path]:
        """Step 10: Generate plots from CSV outputs."""
        step = "10_generate_plots"
        plot_files: Dict[str, Path] = {}

        if self.resume and self.checkpoint.is_complete(step):
            self._log_step(step, "Skipping (already complete)")
            return plot_files

        self._log_step(step, "Generating plots for CSV outputs...")
        start = time.time()

        csv_paths = [Path(p) for p in output_files.values() if str(p).endswith(".csv")]
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

    def _filter_by_qc(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter a dataframe to wells passing QC."""
        if self.qc_well_table is None or self.qc_well_table.empty:
            return df
        if 'qc_pass' not in self.qc_well_table.columns:
            return df
        key_cols = [c for c in ['plate', 'well'] if c in df.columns and c in self.qc_well_table.columns]
        if not key_cols:
            return df
        pass_wells = self.qc_well_table[self.qc_well_table['qc_pass']][key_cols].drop_duplicates()
        return df.merge(pass_wells, on=key_cols, how='inner')

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
            "pipeline_version": "1.0.0",
            "configuration": {
                "config_file": str(self.config_path),
                "genotypes": list(self.config.genotypes.keys()),
                "control_label": self.config.control_label,
                "dilut_column": self.config.dilut_column,
                "n_workers": self.n_workers,
                "qc": self.qc_config.__dict__ if self.qc_config else None,
            },
            "data_summary": {
                "total_cells": len(self.raw_data) if self.raw_data is not None else 0,
                "total_cells_after_qc": (
                    len(self.filtered_raw_data)
                    if self.filtered_raw_data is not None else 0
                ),
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
                "per_genotype": len(self.feature_cols_per_geno),
                "across_all": len(self.feature_cols_across),
            },
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
