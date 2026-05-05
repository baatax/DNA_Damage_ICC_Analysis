"""
DNA Damage Panel Parquet Analysis Pipeline
==========================================

This pipeline processes parquet files generated from DNA Damage Panel analysis,
supporting:
  - Multiple genotypes and drug treatments
  - Batch effect correction via robust z-score normalization
  - Two preprocessing modes: per-genotype and across-all-genotypes
  - Crowding metric compilation per drug/dose
  - Per-well CSV outputs for downstream plotting
  - QC metrics compilation

Input JSON format:
{
    "metadata": {
        "experiment_name": "...",
        "date": "...",
        ...
    },
    "genotypes": {
        "WT": {
            "drugs": {
                "Etoposide": {
                    "path": "/path/to/parquet",
                    "ec50_um": 1.5,
                    "max_dose": "100uM",
                    "moa": "Topoisomerase II inhibitor"
                },
                ...
            }
        },
        ...
    }
}

Author: DNA Damage Analysis Pipeline
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler

warnings.filterwarnings('ignore', category=FutureWarning)


# =============================================================================
# CONFIGURATION & DATA CLASSES
# =============================================================================

@dataclass
class DrugConfig:
    """Configuration for a single drug treatment."""
    path: str
    ec50_um: Optional[float] = None
    max_dose: Optional[str] = None
    moa: Optional[str] = None  # Mechanism of action
    dilution_factor: float = 3.0


@dataclass
class GenotypeConfig:
    """Configuration for a single genotype."""
    drugs: Dict[str, DrugConfig] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    experiment_name: str
    genotypes: Dict[str, GenotypeConfig]
    control_label: str = "DMSO"
    dilut_column: str = "Dilut"
    output_dir: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    qc: Optional["QCConfig"] = None
    normalization: str = "control_based"
    crowding_corr_threshold: float = 0.5
    pca_components: int = 10
    random_seed: int = 42

    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> "ExperimentConfig":
        """Load configuration from JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)

        metadata = data.get("metadata", {})
        experiment_name = metadata.get("experiment_name", "DNA_Damage_Analysis")
        control_label = metadata.get("control_label", "DMSO")
        dilut_column = metadata.get("dilut_column", "Dilut")
        output_dir = metadata.get("output_dir", None)
        qc_data = data.get("qc", metadata.get("qc", {}))
        qc_config = QCConfig.from_dict(qc_data) if qc_data else None

        genotypes = {}
        for geno_name, geno_data in data.get("genotypes", {}).items():
            drugs = {}
            for drug_name, drug_data in geno_data.get("drugs", {}).items():
                drugs[drug_name] = DrugConfig(
                    path=drug_data["path"],
                    ec50_um=drug_data.get("ec50_um"),
                    max_dose=drug_data.get("max_dose"),
                    moa=drug_data.get("moa"),
                    dilution_factor=drug_data.get("dilution_factor", 3.0),
                )
            genotypes[geno_name] = GenotypeConfig(drugs=drugs)

        return cls(
            experiment_name=experiment_name,
            genotypes=genotypes,
            control_label=control_label,
            dilut_column=dilut_column,
            output_dir=output_dir,
            metadata=metadata,
            qc=qc_config,
            normalization=metadata.get("normalization", "control_based"),
            crowding_corr_threshold=metadata.get("crowding_corr_threshold", 0.5),
            pca_components=metadata.get("pca_components", 10),
            random_seed=metadata.get("random_seed", 42),
        )


@dataclass
class QCConfig:
    """Quality control thresholds and warning criteria."""
    n_cells_min: Optional[int] = None
    n_cells_max: Optional[int] = None
    border_max: Optional[float] = None
    coverage_min: Optional[float] = None
    missing_max: Optional[float] = None
    crowding_quantile_low: Optional[float] = None
    crowding_quantile_high: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QCConfig":
        """Create QCConfig from a dictionary."""
        return cls(
            n_cells_min=data.get("n_cells_min"),
            n_cells_max=data.get("n_cells_max"),
            border_max=data.get("border_max"),
            coverage_min=data.get("coverage_min"),
            missing_max=data.get("missing_max"),
            crowding_quantile_low=data.get("crowding_quantile_low"),
            crowding_quantile_high=data.get("crowding_quantile_high"),
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def parse_dilution_string(dilut_str: str) -> float:
    """
    Parse dilution string to numeric molarity in µM.
    
    Handles a wide variety of concentration notations:
    
    Standard units:
        '100uM', '100 uM', '100µM', '100μM' -> 100.0
        '33nM', '33 nM' -> 0.033
        '10pM', '10 pM' -> 0.00001
        '1mM', '1 mM' -> 1000.0
        '0.1M', '100mM' -> 100000.0, 100000.0
        '500fM' -> 0.0000005
        
    Decimal values:
        '0.5uM', '1.5mM', '0.033nM' -> 0.5, 1500.0, 0.000033
        
    Scientific notation:
        '1e-6M' -> 1.0 µM (i.e., 10^-6 M = 1 µM)
        '1E-9M' -> 0.001 µM (i.e., 1 nM)
        '5e2nM' -> 0.5 µM (i.e., 500 nM)
        '1.5e-3mM' -> 1.5 µM (i.e., 0.0015 mM = 1.5 µM)
        
    Alternative formats:
        '100 µmol/L', '33 nmol/L' -> 100.0, 0.033
        
    Controls:
        'DMSO', 'Control', 'Vehicle', 'Untreated', 'Mock', 'PBS' -> 0.0
        '0', '0uM' -> 0.0
        
    Returns:
        float: Concentration in µM, or np.nan if unparseable
    """
    if dilut_str is None:
        return np.nan
    
    # Convert to string and normalize
    dilut_str = str(dilut_str).strip()
    
    # Handle empty string
    if not dilut_str:
        return np.nan
    
    # Normalize unicode micro symbols to 'u'
    # µ (U+00B5 MICRO SIGN) and μ (U+03BC GREEK SMALL LETTER MU)
    dilut_str = dilut_str.replace('µ', 'u').replace('μ', 'u')
    
    # Replace comma with period for European decimal notation
    dilut_str_clean = dilut_str.replace(',', '.').strip()
    
    # Check for control/vehicle labels (case-insensitive)
    control_labels = {
        'dmso', 'control', 'ctrl', 'vehicle', 'untreated', 'mock', 
        'pbs', 'buffer', 'media', 'medium', 'blank', 'neg', 'negative',
        'nt', 'no treatment', 'none', 'water', 'h2o', 'saline'
    }
    if dilut_str_clean.lower() in control_labels:
        return 0.0
    
    # Try to parse as pure number first (assume µM if no unit)
    try:
        val = float(dilut_str_clean)
        return val  # Assume µM
    except ValueError:
        pass
    
    # Unit conversion factors to µM
    unit_to_um = {
        # Molar
        'm': 1e6,
        'mol/l': 1e6,
        'molar': 1e6,
        # Millimolar
        'mm': 1e3,
        'mmol/l': 1e3,
        'millimolar': 1e3,
        # Micromolar (target unit)
        'um': 1.0,
        'umol/l': 1.0,
        'micromolar': 1.0,
        'u': 1.0,  # shorthand
        # Nanomolar
        'nm': 1e-3,
        'nmol/l': 1e-3,
        'nanomolar': 1e-3,
        'n': 1e-3,  # shorthand
        # Picomolar
        'pm': 1e-6,
        'pmol/l': 1e-6,
        'picomolar': 1e-6,
        'p': 1e-6,  # shorthand
        # Femtomolar
        'fm': 1e-9,
        'fmol/l': 1e-9,
        'femtomolar': 1e-9,
        'f': 1e-9,  # shorthand
        # Attomolar (rare but possible)
        'am': 1e-12,
        'amol/l': 1e-12,
        # Percent (w/v or v/v) - approximate
        '%': 1e4,  # 1% ≈ 10mM for many small molecules
    }
    
    # Strategy: Match known unit suffixes (longest first for proper matching)
    # This correctly handles scientific notation like '1.5e-3mM'
    known_units = sorted(unit_to_um.keys(), key=len, reverse=True)
    lower_str = dilut_str_clean.lower()
    
    for unit in known_units:
        # Check if string ends with this unit (allowing optional whitespace)
        pattern = rf'^(.+?)\s*{re.escape(unit)}$'
        match = re.match(pattern, lower_str, re.IGNORECASE)
        if match:
            number_str = match.group(1).strip()
            try:
                value = float(number_str)
                if value >= 0:
                    return value * unit_to_um[unit]
            except ValueError:
                continue  # Try next unit if number parsing fails
    
    # Fallback: general pattern matching
    sci_pattern = r'^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*([a-zA-Z/%][a-zA-Z/]*)?$'
    match = re.match(sci_pattern, dilut_str_clean, re.IGNORECASE)
    
    if match:
        number_str = match.group(1)
        unit = (match.group(2) or 'um').lower()
        
        try:
            value = float(number_str)
            if value < 0:
                return np.nan
            
            multiplier = unit_to_um.get(unit, 1.0)
            return value * multiplier
        except (ValueError, OverflowError):
            return np.nan
    
    return np.nan


def parse_dilution_to_components(dilut_str: str) -> tuple:
    """
    Parse dilution string and return both value and unit separately.
    
    Returns:
        tuple: (numeric_value, unit_string, concentration_in_uM)
               Returns (None, None, np.nan) if unparseable
    
    Example:
        parse_dilution_to_components('100nM') -> (100.0, 'nM', 0.1)
    """
    if dilut_str is None:
        return (None, None, np.nan)
    
    dilut_str = str(dilut_str).strip()
    dilut_str = dilut_str.replace('µ', 'u').replace('μ', 'u')
    
    # Control labels
    if dilut_str.lower() in {'dmso', 'control', 'ctrl', 'vehicle', 'untreated'}:
        return (0.0, 'control', 0.0)
    
    pattern = r'^([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*([a-zA-Z]+(?:/[a-zA-Z]+)?)?$'
    match = re.match(pattern, dilut_str)
    
    if match:
        try:
            value = float(match.group(1))
            unit = match.group(2) or 'uM'
            conc_um = parse_dilution_string(dilut_str)
            return (value, unit, conc_um)
        except ValueError:
            pass
    
    return (None, None, np.nan)


def format_concentration_um(conc_um: float) -> str:
    """Format concentration in µM into a readable unit string."""
    if pd.isna(conc_um):
        return "nan"
    if conc_um == 0:
        return "0uM"
    if conc_um >= 1000:
        return f"{conc_um / 1000:g}mM"
    if conc_um >= 1:
        return f"{conc_um:g}uM"
    if conc_um >= 1e-3:
        return f"{conc_um * 1000:g}nM"
    return f"{conc_um * 1e6:g}pM"


def relabel_pseudo_dilution_series(
    raw_labels: pd.Series,
    control_label: str,
    max_dose: Optional[str],
    dilution_factor: float = 3.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Relabel a pseudo dilution series based on configured top concentration.

    Returns:
        tuple[pd.Series, pd.Series, pd.Series]:
            (corrected_label, corrected_um, dilution_level)
    """
    raw_str = raw_labels.astype(str)
    raw_um = raw_str.apply(parse_dilution_string)
    is_control = raw_str.str.upper() == control_label.upper()

    corrected_um = raw_um.copy()
    corrected_label = raw_str.copy()
    dilution_level = pd.Series(np.nan, index=raw_labels.index, dtype=float)

    max_dose_um = parse_dilution_string(max_dose) if max_dose else np.nan
    if pd.isna(max_dose_um) or max_dose_um <= 0 or dilution_factor <= 0:
        return corrected_label, corrected_um, dilution_level

    non_control_labels = raw_str.loc[~is_control].dropna().unique()
    if len(non_control_labels) == 0:
        return corrected_label, corrected_um, dilution_level

    rank_basis = []
    for label in non_control_labels:
        conc_um = parse_dilution_string(label)
        sort_val = conc_um if not pd.isna(conc_um) else -np.inf
        rank_basis.append((label, sort_val))
    rank_basis = sorted(rank_basis, key=lambda x: x[1], reverse=True)

    label_to_rank = {label: rank for rank, (label, _) in enumerate(rank_basis)}

    for label, rank in label_to_rank.items():
        assigned_um = max_dose_um / (dilution_factor ** rank)
        idx = (raw_str == label) & (~is_control)
        corrected_um.loc[idx] = assigned_um
        corrected_label.loc[idx] = format_concentration_um(assigned_um)
        dilution_level.loc[idx] = float(rank)

    corrected_um.loc[is_control] = 0.0
    corrected_label.loc[is_control] = control_label
    dilution_level.loc[is_control] = np.nan

    return corrected_label, corrected_um, dilution_level


def validate_standard_dilution_scheme(
    doses_um: pd.Series,
    max_dose_um: float = 100.0,
    dilution_factor: float = 3.0,
    n_dilutions: int = 9,
    atol: float = 1e-6,
) -> None:
    """Validate that non-control doses follow the standard 1:3 dilution series."""
    expected = np.array([max_dose_um / (dilution_factor ** i) for i in range(n_dilutions)], dtype=float)
    observed = np.array(sorted(pd.to_numeric(doses_um, errors="coerce").dropna().unique(), reverse=True), dtype=float)
    observed = observed[observed > 0]
    if len(observed) != len(expected):
        raise ValueError(
            f"Invalid dilution scheme: expected {len(expected)} non-control doses, found {len(observed)}."
        )
    if not np.allclose(observed, expected, rtol=0.0, atol=atol):
        raise ValueError(
            "Invalid dilution scheme: observed non-control doses do not match "
            "expected 100uM->... 1:3 series (9 dilutions)."
        )


def get_numeric_columns(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    """Get numeric column names, excluding specified columns."""
    exclude = exclude or []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in exclude]


def robust_zscore(X: np.ndarray, axis: int = 0) -> np.ndarray:
    """Compute robust z-score: (X - median) / MAD."""
    median = np.nanmedian(X, axis=axis, keepdims=True)
    mad = np.nanmedian(np.abs(X - median), axis=axis, keepdims=True)
    mad = np.where(mad == 0, np.nan, mad)
    return (X - median) / mad


# =============================================================================
# DATA LOADING
# =============================================================================

class DNADamageDataLoader:
    """Load and combine DNA damage parquet files based on configuration."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._cache: Dict[str, pd.DataFrame] = {}
    
    def load_single_parquet(
        self,
        genotype: str,
        drug: str,
    ) -> pd.DataFrame:
        """Load a single parquet file and add metadata columns."""
        cache_key = f"{genotype}_{drug}"
        if cache_key in self._cache:
            return self._cache[cache_key].copy()
        
        drug_config = self.config.genotypes[genotype].drugs[drug]
        path = Path(drug_config.path)
        
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found: {path}")
        
        df = pd.read_parquet(path)
        
        # Add metadata columns
        df['genotype'] = genotype
        df['drug'] = drug
        df['moa'] = drug_config.moa
        
        # Parse dilution to numeric
        if self.config.dilut_column in df.columns:
            df['raw_dilut_string'] = df[self.config.dilut_column].astype(str)
            df['dilut_string'] = df['raw_dilut_string'].copy()
            df['dilut_um'] = df['dilut_string'].apply(parse_dilution_string)
            df['is_control'] = df['dilut_string'].str.upper() == self.config.control_label.upper()

            if drug_config.max_dose:
                corrected_label, corrected_um, dilution_level = relabel_pseudo_dilution_series(
                    raw_labels=df['raw_dilut_string'],
                    control_label=self.config.control_label,
                    max_dose=drug_config.max_dose,
                    dilution_factor=drug_config.dilution_factor,
                )
                df['dilut_string'] = corrected_label
                df['dilut_um'] = corrected_um
                df['dilution_level'] = dilution_level
                df['is_control'] = df['raw_dilut_string'].str.upper() == self.config.control_label.upper()

            configured_max_dose_um = parse_dilution_string(drug_config.max_dose) if drug_config.max_dose else 100.0
            if pd.isna(configured_max_dose_um) or configured_max_dose_um <= 0:
                configured_max_dose_um = 100.0
            configured_dilution_factor = drug_config.dilution_factor if drug_config.dilution_factor > 0 else 3.0

            try:
                validate_standard_dilution_scheme(
                    df.loc[~df['is_control'], 'dilut_um'],
                    max_dose_um=configured_max_dose_um,
                    dilution_factor=configured_dilution_factor,
                    n_dilutions=9,
                )
            except ValueError:
                corrected_label, corrected_um, dilution_level = relabel_pseudo_dilution_series(
                    raw_labels=df['raw_dilut_string'],
                    control_label=self.config.control_label,
                    max_dose=drug_config.max_dose or "100uM",
                    dilution_factor=configured_dilution_factor,
                )
                df['dilut_string'] = corrected_label
                df['dilut_um'] = corrected_um
                df['dilution_level'] = dilution_level
                validate_standard_dilution_scheme(
                    df.loc[~df['is_control'], 'dilut_um'],
                    max_dose_um=configured_max_dose_um,
                    dilution_factor=configured_dilution_factor,
                    n_dilutions=9,
                )
        
        # Add EC50 info
        if drug_config.ec50_um is not None:
            df['ec50_um'] = drug_config.ec50_um
            df['log_dose_ec50_ratio'] = np.where(
                df['dilut_um'] > 0,
                np.log10(df['dilut_um'] / drug_config.ec50_um),
                np.nan
            )
        
        self._cache[cache_key] = df
        return df.copy()
    
    def load_genotype(self, genotype: str) -> pd.DataFrame:
        """Load all drug data for a single genotype."""
        dfs = []
        geno_config = self.config.genotypes[genotype]
        
        for drug_name in geno_config.drugs:
            try:
                df = self.load_single_parquet(genotype, drug_name)
                dfs.append(df)
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                continue
        
        if not dfs:
            return pd.DataFrame()
        
        return pd.concat(dfs, ignore_index=True)
    
    def load_all(self) -> pd.DataFrame:
        """Load all data across all genotypes and drugs."""
        dfs = []
        
        for genotype in self.config.genotypes:
            df = self.load_genotype(genotype)
            if not df.empty:
                dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
        
        return pd.concat(dfs, ignore_index=True)


# =============================================================================
# PREPROCESSING & NORMALIZATION
# =============================================================================

class DNADamagePreprocessor:
    """
    Preprocess DNA damage panel data with batch effect correction.
    
    Supports two modes:
      1. Per-genotype: Normalize within each genotype separately
      2. Across-all: Normalize across all genotypes together
    """
    
    # Columns to exclude from feature normalization
    METADATA_COLS = [
        'cell_id', 'plate', 'well', 'site', 'timepoint', 'field_dir', 'base_name',
        'genotype', 'drug', 'moa', 'raw_dilut_string', 'dilut_string', 'dilut_um',
        'dilution_level', 'is_control',
        'ec50_um', 'log_dose_ec50_ratio', 'Dilut', 'group', 'treatment',
        'well_row', 'well_col', 'well_row_index', 'well_col_index',
        'plate_row_edge_dist', 'plate_col_edge_dist', 'plate_edge_dist_min',
        'image_height', 'image_width', 'n_border_cells', 'orig_cell_count',
    ]
    
    # QC metric columns (keep but don't normalize)
    QC_COLS = [
        'cell_count', 'ki67_positive_count', 'foci_total', 'high_damage_cell_count',
        'ki67_threshold', 'segmentation_coverage', 'border_cells_total',
        'orig_cells_total', 'border_fraction_removed',
    ]
    
    def __init__(
        self,
        norm_method: str = "robust_zscore",
        batch_col: str = "plate",
    ):
        """
        Initialize preprocessor.
        
        Parameters
        ----------
        norm_method : str
            Normalization method: 'robust_zscore', 'zscore', or 'none'
        batch_col : str
            Column to use for batch-aware normalization
        """
        self.norm_method = norm_method
        self.batch_col = batch_col
    
    def _get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        """Get feature columns for normalization."""
        exclude = set(self.METADATA_COLS + self.QC_COLS)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return [c for c in numeric_cols if c not in exclude]
    
    def _normalize_block(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
    ) -> pd.DataFrame:
        """Apply normalization to a block of data."""
        df = df.copy()
        
        if self.norm_method == "none" or not feature_cols:
            return df
        
        # Convert feature columns to float32
        df[feature_cols] = df[feature_cols].astype(np.float32)
        X = df[feature_cols].values
        
        if self.norm_method == "robust_zscore":
            X_norm = robust_zscore(X, axis=0)
        elif self.norm_method == "zscore":
            mean = np.nanmean(X, axis=0, keepdims=True)
            std = np.nanstd(X, axis=0, keepdims=True)
            std = np.where(std == 0, np.nan, std)
            X_norm = (X - mean) / std
        else:
            raise ValueError(f"Unknown normalization method: {self.norm_method}")
        
        # Replace inf with nan
        X_norm = np.where(np.isinf(X_norm), np.nan, X_norm)
        df[feature_cols] = X_norm.astype(np.float32)
        
        return df
    
    def preprocess_per_genotype(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Preprocess with per-genotype normalization.
        
        Normalizes within each genotype separately to preserve
        genotype-specific patterns while correcting batch effects.
        """
        feature_cols = self._get_feature_cols(df)
        
        if not feature_cols:
            return df.copy(), []
        
        if 'genotype' not in df.columns:
            # Fall back to global normalization
            return self._normalize_block(df, feature_cols), feature_cols
        
        # Normalize within each genotype
        if self.batch_col and self.batch_col in df.columns:
            # Batch-aware normalization within genotype
            dfs_norm = []
            for geno, geno_df in df.groupby('genotype'):
                geno_norm = geno_df.groupby(self.batch_col, group_keys=False).apply(
                    lambda x: self._normalize_block(x, feature_cols)
                )
                dfs_norm.append(geno_norm)
            df_norm = pd.concat(dfs_norm, ignore_index=True)
        else:
            # Simple per-genotype normalization
            df_norm = df.groupby('genotype', group_keys=False).apply(
                lambda x: self._normalize_block(x, feature_cols)
            )
        
        return df_norm, feature_cols
    
    def preprocess_across_all(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Preprocess with across-all-genotypes normalization.
        
        Normalizes across all data together, useful for comparing
        genotypes directly.
        """
        feature_cols = self._get_feature_cols(df)
        
        if not feature_cols:
            return df.copy(), []
        
        if self.batch_col and self.batch_col in df.columns:
            # Batch-aware normalization across all data
            df_norm = df.groupby(self.batch_col, group_keys=False).apply(
                lambda x: self._normalize_block(x, feature_cols)
            )
        else:
            # Global normalization
            df_norm = self._normalize_block(df, feature_cols)
        
        return df_norm, feature_cols


# =============================================================================
# CROWDING METRICS
# =============================================================================

class CrowdingAnalyzer:
    """
    Compile crowding metrics per drug/dose across all cells.
    
    Crowding metrics typically include:
      - crowding_local_mean: Mean local cell density
      - nn_dist_px: Nearest neighbor distance
      - nbr_count_r*: Neighbor counts at various radii
    """
    
    CROWDING_COLS = [
        'crowding_local_mean', 'crowding_local_std', 'crowding_local_max',
        'nn_dist_px', 'mean_k1_dist_px', 'mean_k3_dist_px', 'mean_k5_dist_px',
        'nbr_count_r50', 'nbr_count_r100',
        'local_mean_area_k1', 'local_mean_area_k3', 'local_mean_area_k5',
    ]
    
    def __init__(
        self,
        dilut_column: str = "Dilut",
        control_label: str = "DMSO",
    ):
        self.dilut_column = dilut_column
        self.control_label = control_label
    
    def _get_available_crowding_cols(self, df: pd.DataFrame) -> List[str]:
        """Get crowding columns that exist in the dataframe."""
        return [c for c in self.CROWDING_COLS if c in df.columns]
    
    def compute_crowding_summary(
        self,
        df: pd.DataFrame,
        groupby: List[str],
    ) -> pd.DataFrame:
        """
        Compute summary statistics for crowding metrics.
        
        Parameters
        ----------
        df : pd.DataFrame
            Single-cell data with crowding features
        groupby : list
            Columns to group by (e.g., ['drug', 'dilut_string'])
        
        Returns
        -------
        pd.DataFrame
            Summary statistics per group
        """
        crowding_cols = self._get_available_crowding_cols(df)
        
        if not crowding_cols:
            print("Warning: No crowding columns found in data")
            return pd.DataFrame()
        
        # Compute aggregations
        agg_funcs = {
            col: ['mean', 'median', 'std', 'min', 'max', 'count']
            for col in crowding_cols
        }
        
        summary = df.groupby(groupby).agg(agg_funcs)
        
        # Flatten column names
        summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
        summary = summary.reset_index()
        
        # Add cell count
        cell_counts = df.groupby(groupby).size().rename('total_cells')
        summary = summary.merge(cell_counts.reset_index(), on=groupby)
        
        return summary
    
    def compile_crowding_by_drug_dose(
        self,
        df: pd.DataFrame,
        dataset_label: str,
    ) -> pd.DataFrame:
        """
        Compile crowding metrics for every drug/dose combination.
        
        Parameters
        ----------
        df : pd.DataFrame
            Single-cell data
        dataset_label : str
            Label for the dataset (e.g., 'per_genotype', 'across_all')
        
        Returns
        -------
        pd.DataFrame
            Crowding summary with dataset label
        """
        groupby_cols = []
        
        # Build groupby columns based on what's available
        if 'genotype' in df.columns:
            groupby_cols.append('genotype')
        if 'drug' in df.columns:
            groupby_cols.append('drug')
        if 'dilut_string' in df.columns:
            groupby_cols.append('dilut_string')
        elif self.dilut_column in df.columns:
            groupby_cols.append(self.dilut_column)
        
        if not groupby_cols:
            print("Warning: No groupby columns available for crowding analysis")
            return pd.DataFrame()
        
        summary = self.compute_crowding_summary(df, groupby_cols)
        summary['dataset'] = dataset_label
        
        # Add numeric dose for sorting
        dilut_col = 'dilut_string' if 'dilut_string' in summary.columns else self.dilut_column
        if dilut_col in summary.columns:
            summary['dilut_um'] = summary[dilut_col].apply(parse_dilution_string)
            summary['is_control'] = summary[dilut_col].str.upper() == self.control_label.upper()

        return summary

    def compute_crowding_exclusions(
        self,
        profiles: pd.DataFrame,
        feature_cols: List[str],
        threshold: float = 0.5,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Identify features highly correlated with a crowding proxy.

        Computes a per-well crowding proxy from available crowding columns
        (falls back to ``cell_count``), then returns a correlation table and
        the list of features exceeding ``|corr| > threshold``.

        Parameters
        ----------
        profiles : pd.DataFrame
            Well-level profiles containing feature columns and optionally
            crowding columns (``crowding_local_mean``, ``crowding_density``,
            ``crowding_sigma50``, ``crowding``, ``cell_count``).
        feature_cols : list[str]
            Feature columns to evaluate.
        threshold : float
            Absolute correlation above which a feature is excluded.

        Returns
        -------
        corr_df : pd.DataFrame
            Per-feature correlation with the crowding proxy.
        excluded : list[str]
            Feature names exceeding the threshold.
        """
        proxy_candidates = [
            'crowding_local_mean', 'crowding_density',
            'crowding_sigma50', 'crowding', 'cell_count',
        ]
        proxy_col = None
        for c in proxy_candidates:
            if c in profiles.columns and profiles[c].notna().sum() > 2:
                proxy_col = c
                break

        if proxy_col is None:
            empty = pd.DataFrame(columns=['feature', 'crowding_corr'])
            return empty, []

        proxy = profiles[proxy_col].astype(float)
        records = []
        for feat in feature_cols:
            if feat not in profiles.columns:
                continue
            vals = profiles[feat].astype(float)
            mask = vals.notna() & proxy.notna()
            if mask.sum() < 3:
                continue
            corr = vals[mask].corr(proxy[mask])
            records.append({'feature': feat, 'crowding_corr': corr})

        corr_df = pd.DataFrame(records)
        if corr_df.empty:
            return corr_df, []

        excluded = corr_df.loc[
            corr_df['crowding_corr'].abs() > threshold, 'feature'
        ].tolist()
        return corr_df, excluded


# =============================================================================
# QC METRICS
# =============================================================================

class QCMetricsCompiler:
    """Compile quality control metrics per well."""
    
    # Per-well QC metrics to extract
    WELL_QC_METRICS = [
        'cell_count', 'cell_area_mean', 'segmentation_coverage',
        'border_cells_total', 'orig_cells_total', 'border_fraction_removed',
        'ki67_positive_count', 'ki67_positive_fraction',
        'foci_mean_per_cell', 'foci_total', 'high_damage_cell_count',
        'high_damage_fraction',
    ]
    
    # Per-cell QC-relevant metrics for aggregation
    CELL_QC_METRICS = [
        'area', 'perimeter', 'eccentricity', 'solidity',
        'crowding_local_mean', 'nn_dist_px',
    ]
    
    def compile_well_qc(
        self,
        df: pd.DataFrame,
        groupby_cols: List[str] = None,
    ) -> pd.DataFrame:
        """
        Compile QC metrics per well.
        
        Parameters
        ----------
        df : pd.DataFrame
            Single-cell data
        groupby_cols : list, optional
            Columns to group by. Default: ['plate', 'well']
        
        Returns
        -------
        pd.DataFrame
            Well-level QC metrics
        """
        if groupby_cols is None:
            groupby_cols = ['plate', 'well']
        
        # Filter to available columns
        groupby_cols = [c for c in groupby_cols if c in df.columns]
        if not groupby_cols:
            groupby_cols = ['well'] if 'well' in df.columns else []
        
        if not groupby_cols:
            print("Warning: No groupby columns for QC compilation")
            return pd.DataFrame()
        
        # Aggregate cell-level metrics
        cell_metrics = [c for c in self.CELL_QC_METRICS if c in df.columns]
        feature_cols = self._get_feature_cols(df)
        
        qc_data = {}
        grouped = df.groupby(groupby_cols)
        
        # Cell count
        qc_data['cell_count'] = grouped.size()
        
        # Mean/std for cell-level metrics
        for metric in cell_metrics:
            qc_data[f'{metric}_mean'] = grouped[metric].mean()
            qc_data[f'{metric}_std'] = grouped[metric].std()
            qc_data[f'{metric}_median'] = grouped[metric].median()
        
        # Special metrics
        if 'ki67_positive' in df.columns:
            qc_data['ki67_positive_count'] = grouped['ki67_positive'].sum()
            qc_data['ki67_positive_fraction'] = (
                qc_data['ki67_positive_count'] / qc_data['cell_count']
            )
        
        if 'foci_count' in df.columns:
            qc_data['foci_mean_per_cell'] = grouped['foci_count'].mean()
            qc_data['foci_std_per_cell'] = grouped['foci_count'].std()
            qc_data['foci_total'] = grouped['foci_count'].sum()
            qc_data['cells_with_foci_count'] = grouped['foci_count'].apply(
                lambda x: (x > 0).sum()
            )
            qc_data['cells_with_foci_fraction'] = (
                qc_data['cells_with_foci_count'] / qc_data['cell_count']
            )
            qc_data['high_damage_count'] = grouped['foci_count'].apply(
                lambda x: (x > 5).sum()
            )
            qc_data['high_damage_fraction'] = (
                qc_data['high_damage_count'] / qc_data['cell_count']
            )
        
        # Combine into dataframe
        qc_df = pd.DataFrame(qc_data).reset_index()

        if feature_cols:
            missing_fraction = df[feature_cols].isna().mean(axis=1)
            qc_df['missing_feature_fraction'] = grouped.apply(
                lambda x: missing_fraction.loc[x.index].mean()
            ).values

        if 'border_cells_total' in qc_df.columns and 'orig_cells_total' in qc_df.columns:
            qc_df['fraction_border_cells'] = (
                qc_df['border_cells_total'] / qc_df['orig_cells_total']
            )
        
        # Add metadata if available
        meta_cols = ['genotype', 'drug', 'dilut_string', 'moa']
        for col in meta_cols:
            if col in df.columns:
                meta = grouped[col].first()
                qc_df = qc_df.merge(meta.reset_index(), on=groupby_cols, how='left')
        
        return qc_df

    def apply_qc_rules(
        self,
        qc_df: pd.DataFrame,
        qc_config: Optional["QCConfig"],
    ) -> pd.DataFrame:
        """Apply QC thresholds and return decision table with reasons."""
        if qc_df.empty:
            return qc_df

        qc_df = qc_df.copy()
        qc_df['qc_pass'] = True
        qc_df['qc_fail_reasons'] = ""
        qc_df['qc_warning_reasons'] = ""

        if qc_config is None:
            return qc_df

        def _append_reason(series: pd.Series, reason: str) -> pd.Series:
            return series.where(series == "", series + "; ") + reason

        if qc_config.n_cells_min is not None and 'cell_count' in qc_df.columns:
            fail = qc_df['cell_count'] < qc_config.n_cells_min
            qc_df.loc[fail, 'qc_pass'] = False
            qc_df.loc[fail, 'qc_fail_reasons'] = _append_reason(
                qc_df.loc[fail, 'qc_fail_reasons'], f"n_cells<{qc_config.n_cells_min}"
            )

        if qc_config.n_cells_max is not None and 'cell_count' in qc_df.columns:
            fail = qc_df['cell_count'] > qc_config.n_cells_max
            qc_df.loc[fail, 'qc_pass'] = False
            qc_df.loc[fail, 'qc_fail_reasons'] = _append_reason(
                qc_df.loc[fail, 'qc_fail_reasons'], f"n_cells>{qc_config.n_cells_max}"
            )

        if qc_config.border_max is not None and 'fraction_border_cells' in qc_df.columns:
            fail = qc_df['fraction_border_cells'] > qc_config.border_max
            qc_df.loc[fail, 'qc_pass'] = False
            qc_df.loc[fail, 'qc_fail_reasons'] = _append_reason(
                qc_df.loc[fail, 'qc_fail_reasons'], f"border_fraction>{qc_config.border_max}"
            )

        if qc_config.coverage_min is not None and 'segmentation_coverage' in qc_df.columns:
            fail = qc_df['segmentation_coverage'] < qc_config.coverage_min
            qc_df.loc[fail, 'qc_pass'] = False
            qc_df.loc[fail, 'qc_fail_reasons'] = _append_reason(
                qc_df.loc[fail, 'qc_fail_reasons'], f"coverage<{qc_config.coverage_min}"
            )

        if qc_config.missing_max is not None and 'missing_feature_fraction' in qc_df.columns:
            fail = qc_df['missing_feature_fraction'] > qc_config.missing_max
            qc_df.loc[fail, 'qc_pass'] = False
            qc_df.loc[fail, 'qc_fail_reasons'] = _append_reason(
                qc_df.loc[fail, 'qc_fail_reasons'], f"missing>{qc_config.missing_max}"
            )

        warn_low = qc_config.crowding_quantile_low
        warn_high = qc_config.crowding_quantile_high
        if warn_low is not None or warn_high is not None:
            crowding_col = next(
                (c for c in qc_df.columns if c.startswith("crowding_local_mean_")), None
            )
            if crowding_col:
                values = qc_df[crowding_col].dropna()
                if not values.empty:
                    low_val = values.quantile(warn_low) if warn_low is not None else None
                    high_val = values.quantile(warn_high) if warn_high is not None else None
                    if low_val is not None:
                        warn = qc_df[crowding_col] < low_val
                        qc_df.loc[warn, 'qc_warning_reasons'] = _append_reason(
                            qc_df.loc[warn, 'qc_warning_reasons'], "crowding_low"
                        )
                    if high_val is not None:
                        warn = qc_df[crowding_col] > high_val
                        qc_df.loc[warn, 'qc_warning_reasons'] = _append_reason(
                            qc_df.loc[warn, 'qc_warning_reasons'], "crowding_high"
                        )

        return qc_df

    def _get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        exclude = {
            'cell_id', 'plate', 'well', 'site', 'timepoint', 'field_dir', 'base_name',
            'genotype', 'drug', 'moa', 'raw_dilut_string', 'dilut_string', 'dilut_um',
            'dilution_level', 'is_control',
            'ec50_um', 'log_dose_ec50_ratio', 'Dilut', 'group', 'treatment',
            'well_row', 'well_col', 'well_row_index', 'well_col_index',
            'plate_row_edge_dist', 'plate_col_edge_dist', 'plate_edge_dist_min',
            'image_height', 'image_width', 'n_border_cells', 'orig_cell_count',
        }
        qc_cols = set(self.WELL_QC_METRICS)
        qc_cols.update({'cell_count'})
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return [c for c in numeric_cols if c not in exclude and c not in qc_cols]


# =============================================================================
# FEATURE SELECTION
# =============================================================================

class FeatureSelector:
    """Select informative features for downstream analysis."""
    
    def __init__(
        self,
        var_thresh: float = 1e-5,
        corr_thresh: float = 0.95,
        nan_thresh: float = 0.5,
    ):
        """
        Parameters
        ----------
        var_thresh : float
            Minimum variance threshold
        corr_thresh : float
            Maximum correlation threshold for redundancy removal
        nan_thresh : float
            Maximum fraction of NaN values allowed
        """
        self.var_thresh = var_thresh
        self.corr_thresh = corr_thresh
        self.nan_thresh = nan_thresh
    
    def select_features(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
    ) -> List[str]:
        """
        Select features based on variance and correlation.
        
        Returns list of selected feature names.
        """
        if not feature_cols:
            return []
        
        # Filter to existing columns
        feature_cols = [c for c in feature_cols if c in df.columns]
        if not feature_cols:
            return []
        
        X = df[feature_cols]
        
        # 1. Remove high-NaN columns
        nan_frac = X.isna().mean()
        keep_nan = nan_frac[nan_frac <= self.nan_thresh].index.tolist()
        
        if not keep_nan:
            return []
        
        # 2. Variance filter
        variances = X[keep_nan].var()
        keep_var = variances[variances > self.var_thresh].index.tolist()
        
        if not keep_var:
            return []
        
        # 3. Correlation filter
        corr = X[keep_var].corr().abs()
        to_drop = set()
        
        for col in corr.columns:
            if col in to_drop:
                continue
            high_corr = corr.index[(corr[col] > self.corr_thresh) & (corr.index != col)]
            to_drop.update(high_corr)
        
        selected = [c for c in keep_var if c not in to_drop]
        return selected
