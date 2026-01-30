#!/usr/bin/env python3
"""
DNA Damage Panel Analysis Pipeline - Usage Examples
====================================================

This script demonstrates how to use the DNA damage parquet analysis pipeline.
It includes:
  1. Creating synthetic test data
  2. Running the full pipeline
  3. Performing dose-response analysis
  4. Generating summary reports

Run this script to test the pipeline with synthetic data.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List

# Import pipeline components
from dna_damage_parquet_pipeline import (
    ExperimentConfig,
    DNADamageAnalysisPipeline,
    DNADamageDataLoader,
    DNADamagePreprocessor,
    CrowdingAnalyzer,
    QCMetricsCompiler,
    run_pipeline_from_json,
    create_example_config,
    parse_dilution_string,
)

from dose_response_analysis import (
    DoseResponseAnalyzer,
    StatisticalComparator,
    ResponseSummarizer,
    analyze_dose_response,
    fit_dose_response,
)


# =============================================================================
# SYNTHETIC DATA GENERATION
# =============================================================================

def generate_synthetic_dna_damage_data(
    n_cells_per_well: int = 100,
    n_wells_per_dose: int = 3,
    genotype: str = "WT",
    drug: str = "Etoposide",
    ec50_um: float = 1.5,
    max_dose_um: float = 100.0,
    dilution_factor: float = 3.0,
    n_doses: int = 8,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic DNA damage panel data for testing.
    
    This creates realistic-looking single-cell data with:
      - Dose-dependent foci counts
      - Ki67 proliferation marker
      - Crowding metrics
      - QC features
    """
    np.random.seed(random_seed)
    
    # Generate dose series
    doses_um = [max_dose_um / (dilution_factor ** i) for i in range(n_doses)]
    doses_um.append(0.0)  # DMSO control
    
    dose_strings = []
    for d in doses_um:
        if d == 0:
            dose_strings.append("DMSO")
        elif d >= 1000:
            dose_strings.append(f"{d/1000:.0f}mM")
        elif d >= 1:
            dose_strings.append(f"{d:.0f}uM")
        elif d >= 0.001:
            dose_strings.append(f"{d*1000:.0f}nM")
        else:
            dose_strings.append(f"{d*1e6:.0f}pM")
    
    rows = []
    cell_id = 0
    
    for dose_idx, (dose_um, dose_str) in enumerate(zip(doses_um, dose_strings)):
        for well_idx in range(n_wells_per_dose):
            well_row = chr(ord('A') + (dose_idx * n_wells_per_dose + well_idx) // 12)
            well_col = (dose_idx * n_wells_per_dose + well_idx) % 12 + 1
            well = f"{well_row}{well_col:02d}"
            
            for _ in range(n_cells_per_well):
                # Dose-response relationship for foci
                if dose_um > 0:
                    # Hill equation response
                    response = 1 / (1 + (ec50_um / dose_um) ** 1.5)
                    base_foci = 1 + response * 25  # 1-26 foci range
                else:
                    base_foci = 1.0
                
                foci_count = max(0, int(np.random.poisson(base_foci)))
                
                # Ki67 decreases with damage
                ki67_prob = 0.7 - 0.5 * (foci_count / 30)
                ki67_positive = int(np.random.random() < max(0.1, ki67_prob))
                
                # Morphological features
                area = np.random.lognormal(8.5, 0.3)  # ~5000 px^2
                perimeter = np.sqrt(area) * np.random.uniform(3.5, 4.5)
                eccentricity = np.random.beta(2, 5)
                solidity = np.random.beta(20, 2)
                
                # Intensity features (normalized 0-1)
                dapi_mean = np.random.beta(3, 2) * 0.8 + 0.1
                gamma_h2ax_mean = np.clip(
                    np.random.normal(0.1 + 0.03 * foci_count, 0.05), 0, 1
                )
                ki67_mean = np.random.beta(2 + ki67_positive * 3, 3)
                syto_rna_mean = np.random.beta(3, 2) * 0.6 + 0.2
                
                # Crowding metrics
                nn_dist = np.random.exponential(50)
                crowding_local_mean = np.random.exponential(5)
                nbr_count_r50 = np.random.poisson(3)
                nbr_count_r100 = np.random.poisson(8)
                
                rows.append({
                    'cell_id': cell_id,
                    'plate': f"{genotype}_{drug}_Plate_001",
                    'well': well,
                    'site': 's1',
                    'Dilut': dose_str,
                    
                    # DNA damage features
                    'foci_count': foci_count,
                    'foci_total_intensity': foci_count * np.random.uniform(0.5, 1.5),
                    'foci_mean_area': np.random.lognormal(3, 0.5) if foci_count > 0 else 0,
                    
                    # Ki67 features
                    'ki67_positive': ki67_positive,
                    'ki67_mean_intensity': ki67_mean,
                    
                    # Morphology
                    'area': area,
                    'perimeter': perimeter,
                    'eccentricity': eccentricity,
                    'solidity': solidity,
                    'circularity': 4 * np.pi * area / (perimeter ** 2),
                    
                    # Intensity features
                    'dapi_mean_intensity': dapi_mean,
                    'gamma_h2ax_mean_intensity': gamma_h2ax_mean,
                    'syto_rna_mean_intensity': syto_rna_mean,
                    
                    # Crowding
                    'nn_dist_px': nn_dist,
                    'crowding_local_mean': crowding_local_mean,
                    'nbr_count_r50': nbr_count_r50,
                    'nbr_count_r100': nbr_count_r100,
                    'mean_k3_dist_px': nn_dist * np.random.uniform(1.2, 1.8),
                })
                cell_id += 1
    
    return pd.DataFrame(rows)


def create_test_dataset(output_dir: Path) -> Dict[str, Path]:
    """
    Create a complete test dataset with multiple genotypes and drugs.
    
    Returns dict mapping (genotype, drug) to parquet file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define test conditions
    conditions = {
        'WT': {
            'Etoposide': {'ec50': 1.5, 'max_dose': 100},
            'Camptothecin': {'ec50': 0.5, 'max_dose': 33},
            'Cisplatin': {'ec50': 5.0, 'max_dose': 100},
        },
        'BRCA1_KO': {
            'Etoposide': {'ec50': 0.8, 'max_dose': 100},
            'Camptothecin': {'ec50': 0.2, 'max_dose': 33},
            'Cisplatin': {'ec50': 1.5, 'max_dose': 100},
        },
        'p53_KO': {
            'Etoposide': {'ec50': 2.5, 'max_dose': 100},
            'Camptothecin': {'ec50': 0.8, 'max_dose': 33},
        },
    }
    
    paths = {}
    
    for genotype, drugs in conditions.items():
        for drug, params in drugs.items():
            print(f"  Generating {genotype} + {drug}...")
            
            df = generate_synthetic_dna_damage_data(
                n_cells_per_well=50,  # Smaller for testing
                n_wells_per_dose=2,
                genotype=genotype,
                drug=drug,
                ec50_um=params['ec50'],
                max_dose_um=params['max_dose'],
                random_seed=hash(f"{genotype}_{drug}") % 2**31,
            )
            
            filename = f"{genotype}_{drug}_single_cell_features.parquet"
            filepath = output_dir / filename
            df.to_parquet(filepath, index=False)
            
            paths[(genotype, drug)] = filepath
    
    return paths


def create_test_config(parquet_paths: Dict, output_path: Path) -> Path:
    """Create a JSON config file for the test dataset."""
    
    config = {
        "metadata": {
            "experiment_name": "Test_DNA_Damage_Analysis",
            "date": "2024-01-01",
            "control_label": "DMSO",
            "dilut_column": "Dilut",
            "output_dir": str(output_path.parent / "analysis_results"),
        },
        "genotypes": {}
    }
    
    # Organize paths by genotype
    genotypes = {}
    for (geno, drug), path in parquet_paths.items():
        if geno not in genotypes:
            genotypes[geno] = {}
        genotypes[geno][drug] = path
    
    # Build config structure
    ec50_map = {
        ('WT', 'Etoposide'): 1.5,
        ('WT', 'Camptothecin'): 0.5,
        ('WT', 'Cisplatin'): 5.0,
        ('BRCA1_KO', 'Etoposide'): 0.8,
        ('BRCA1_KO', 'Camptothecin'): 0.2,
        ('BRCA1_KO', 'Cisplatin'): 1.5,
        ('p53_KO', 'Etoposide'): 2.5,
        ('p53_KO', 'Camptothecin'): 0.8,
    }
    
    for geno, drugs in genotypes.items():
        config["genotypes"][geno] = {"drugs": {}}
        for drug, path in drugs.items():
            config["genotypes"][geno]["drugs"][drug] = {
                "path": str(path),
                "ec50_um": ec50_map.get((geno, drug), 1.0),
                "moa": {
                    "Etoposide": "Topoisomerase II inhibitor",
                    "Camptothecin": "Topoisomerase I inhibitor",
                    "Cisplatin": "DNA crosslinker",
                }.get(drug, "Unknown"),
            }
    
    config_path = output_path
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config_path


# =============================================================================
# EXAMPLE USAGE FUNCTIONS
# =============================================================================

def example_basic_pipeline():
    """Example: Run the basic pipeline from a JSON config."""
    print("\n" + "="*70)
    print("EXAMPLE: Basic Pipeline Execution")
    print("="*70)
    
    # Create test data directory
    test_dir = Path("./test_dna_damage_data")
    test_dir.mkdir(exist_ok=True)
    
    print("\n1. Generating synthetic test data...")
    parquet_paths = create_test_dataset(test_dir / "parquets")
    
    print("\n2. Creating configuration file...")
    config_path = create_test_config(parquet_paths, test_dir / "test_config.json")
    print(f"   Config saved to: {config_path}")
    
    print("\n3. Running analysis pipeline...")
    results = run_pipeline_from_json(config_path)
    
    print("\n4. Pipeline complete! Output files:")
    for name, path in results.items():
        if isinstance(path, dict):
            print(f"   {name}:")
            for k, v in list(path.items())[:3]:
                print(f"      {k}: {v}")
            if len(path) > 3:
                print(f"      ... and {len(path)-3} more files")
        else:
            print(f"   {name}: {path}")
    
    return results


def example_dose_response_analysis():
    """Example: Perform dose-response analysis on existing data."""
    print("\n" + "="*70)
    print("EXAMPLE: Dose-Response Analysis")
    print("="*70)
    
    # Generate sample data
    print("\n1. Generating sample dose-response data...")
    df = generate_synthetic_dna_damage_data(
        genotype="WT",
        drug="Etoposide",
        ec50_um=1.5,
        n_cells_per_well=100,
    )
    
    # Add computed columns
    df['dilut_um'] = df['Dilut'].apply(parse_dilution_string)
    df['is_control'] = df['Dilut'].str.upper() == 'DMSO'
    df['genotype'] = 'WT'
    df['drug'] = 'Etoposide'
    
    print(f"   Generated {len(df)} cells across {df['Dilut'].nunique()} doses")
    
    # Run dose-response analysis
    print("\n2. Fitting dose-response curves...")
    analyzer = DoseResponseAnalyzer(dose_column='dilut_um')
    
    # Fit for foci_count
    fits = analyzer.fit_drug_response(
        df, 
        'foci_count',
        groupby=['genotype', 'drug']
    )
    
    print("\n   Fit results:")
    print(f"   EC50: {fits['ec50'].values[0]:.3f} uM")
    print(f"   Hill slope: {fits['hill_slope'].values[0]:.2f}")
    print(f"   R-squared: {fits['r_squared'].values[0]:.3f}")
    
    # Compute summary by dose
    print("\n3. Computing dose-response summary...")
    summarizer = ResponseSummarizer()
    summary = summarizer.summarize_by_dose(
        df,
        response_cols=['foci_count', 'ki67_positive'],
        groupby=['genotype', 'drug']
    )
    
    print("\n   Summary by dose (first 5 rows):")
    print(summary[['dilut_um', 'foci_count_mean', 'foci_count_std', 
                   'ki67_positive_mean']].head().to_string(index=False))
    
    # Fold-change analysis
    print("\n4. Computing fold-change vs control...")
    fc = summarizer.compute_fold_change_vs_control(
        df,
        response_cols=['foci_count', 'gamma_h2ax_mean_intensity'],
        groupby=['genotype', 'drug']
    )
    
    print("\n   Fold-change (first 5 rows):")
    cols_to_show = ['dilut_um', 'foci_count_fold_change', 
                    'gamma_h2ax_mean_intensity_fold_change']
    cols_to_show = [c for c in cols_to_show if c in fc.columns]
    print(fc[cols_to_show].head().to_string(index=False))
    
    return {
        'data': df,
        'fits': fits,
        'summary': summary,
        'fold_change': fc,
    }


def example_genotype_comparison():
    """Example: Compare responses between genotypes."""
    print("\n" + "="*70)
    print("EXAMPLE: Genotype Comparison")
    print("="*70)
    
    # Generate data for multiple genotypes
    print("\n1. Generating multi-genotype data...")
    
    dfs = []
    genotype_ec50s = {
        'WT': 1.5,
        'BRCA1_KO': 0.5,  # More sensitive
        'p53_KO': 2.5,    # Less sensitive
    }
    
    for geno, ec50 in genotype_ec50s.items():
        df = generate_synthetic_dna_damage_data(
            genotype=geno,
            drug="Etoposide",
            ec50_um=ec50,
            n_cells_per_well=80,
            random_seed=hash(geno) % 2**31,
        )
        df['dilut_um'] = df['Dilut'].apply(parse_dilution_string)
        df['is_control'] = df['Dilut'].str.upper() == 'DMSO'
        df['genotype'] = geno
        df['drug'] = 'Etoposide'
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"   Generated {len(combined)} cells across {combined['genotype'].nunique()} genotypes")
    
    # Fit dose-response for each genotype
    print("\n2. Fitting dose-response curves per genotype...")
    analyzer = DoseResponseAnalyzer()
    fits = analyzer.fit_drug_response(
        combined,
        'foci_count',
        groupby=['genotype', 'drug']
    )
    
    print("\n   EC50 by genotype:")
    for _, row in fits.iterrows():
        print(f"   {row['genotype']}: EC50 = {row['ec50']:.3f} uM (R^2 = {row['r_squared']:.3f})")
    
    # Statistical comparison
    print("\n3. Comparing EC50 values between genotypes...")
    comparator = StatisticalComparator()
    comparisons = comparator.compare_ec50s(fits, groupby='genotype')
    
    if not comparisons.empty:
        print("\n   EC50 comparisons:")
        for _, row in comparisons.iterrows():
            print(f"   {row['genotype_1']} vs {row['genotype_2']}: "
                  f"ratio = {row['ec50_ratio']:.2f}x, p = {row['p_value']:.4f}")
    
    # Compare responses at specific dose
    print("\n4. Comparing responses at 10 uM...")
    dose_comparison = comparator.compare_responses_at_dose(
        combined,
        'foci_count',
        dose_col='dilut_um',
        groupby='genotype',
        drug_col='drug'
    )
    
    if not dose_comparison.empty:
        print("\n   Response comparisons at specific doses:")
        for _, row in dose_comparison.head(3).iterrows():
            print(f"   {row['genotype_1']} vs {row['genotype_2']} at {row['dilut_um']:.1f} uM: "
                  f"p = {row['p_value']:.4f}")
    
    return {
        'data': combined,
        'fits': fits,
        'ec50_comparisons': comparisons,
        'dose_comparisons': dose_comparison,
    }


def example_crowding_analysis():
    """Example: Analyze crowding metrics."""
    print("\n" + "="*70)
    print("EXAMPLE: Crowding Metrics Analysis")
    print("="*70)
    
    # Generate data
    print("\n1. Generating data with crowding metrics...")
    df = generate_synthetic_dna_damage_data(
        genotype="WT",
        drug="Etoposide",
        n_cells_per_well=100,
    )
    df['dilut_um'] = df['Dilut'].apply(parse_dilution_string)
    df['dilut_string'] = df['Dilut']
    df['genotype'] = 'WT'
    df['drug'] = 'Etoposide'
    
    print(f"   Generated {len(df)} cells")
    
    # Analyze crowding
    print("\n2. Computing crowding metrics by dose...")
    crowding = CrowdingAnalyzer()
    summary = crowding.compile_crowding_by_drug_dose(df, "test_dataset")
    
    print("\n   Crowding summary columns:")
    print(f"   {list(summary.columns)[:10]}...")
    
    print("\n   Mean crowding by dose (first 5 doses):")
    cols = ['dilut_string', 'total_cells']
    if 'crowding_local_mean_mean' in summary.columns:
        cols.append('crowding_local_mean_mean')
    if 'nn_dist_px_mean' in summary.columns:
        cols.append('nn_dist_px_mean')
    
    print(summary[cols].head().to_string(index=False))
    
    return summary


def example_qc_compilation():
    """Example: Compile QC metrics."""
    print("\n" + "="*70)
    print("EXAMPLE: QC Metrics Compilation")
    print("="*70)
    
    # Generate data
    print("\n1. Generating data...")
    df = generate_synthetic_dna_damage_data(
        genotype="WT",
        drug="Etoposide",
        n_cells_per_well=100,
    )
    df['genotype'] = 'WT'
    df['drug'] = 'Etoposide'
    df['dilut_string'] = df['Dilut']
    
    print(f"   Generated {len(df)} cells across {df['well'].nunique()} wells")
    
    # Compile QC
    print("\n2. Compiling QC metrics per well...")
    qc = QCMetricsCompiler()
    qc_metrics = qc.compile_well_qc(
        df,
        groupby_cols=['plate', 'well', 'genotype', 'drug', 'dilut_string']
    )
    
    print("\n   QC metrics columns:")
    print(f"   {list(qc_metrics.columns)}")
    
    print("\n   QC summary (first 5 wells):")
    cols = ['well', 'cell_count', 'area_mean', 'foci_mean_per_cell']
    cols = [c for c in cols if c in qc_metrics.columns]
    print(qc_metrics[cols].head().to_string(index=False))
    
    # Summary statistics
    print("\n   Overall QC summary:")
    print(f"   Total wells: {len(qc_metrics)}")
    print(f"   Mean cells/well: {qc_metrics['cell_count'].mean():.1f}")
    if 'foci_mean_per_cell' in qc_metrics.columns:
        print(f"   Mean foci/cell: {qc_metrics['foci_mean_per_cell'].mean():.2f}")
    
    return qc_metrics


# =============================================================================
# MAIN
# =============================================================================

def run_all_examples():
    """Run all example functions."""
    print("\n" + "#"*70)
    print("# DNA DAMAGE PANEL ANALYSIS PIPELINE - DEMONSTRATION")
    print("#"*70)
    
    # Run examples
    example_dose_response_analysis()
    example_genotype_comparison()
    example_crowding_analysis()
    example_qc_compilation()
    
    # Full pipeline (generates files)
    print("\n" + "="*70)
    print("Running full pipeline example (this will create output files)...")
    print("="*70)
    
    response = input("\nRun full pipeline with file output? (y/n): ").strip().lower()
    if response == 'y':
        example_basic_pipeline()
    else:
        print("Skipped full pipeline execution.")
    
    print("\n" + "#"*70)
    print("# DEMONSTRATION COMPLETE")
    print("#"*70)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        example_name = sys.argv[1].lower()
        
        if example_name == "basic":
            example_basic_pipeline()
        elif example_name == "dose-response":
            example_dose_response_analysis()
        elif example_name == "genotype":
            example_genotype_comparison()
        elif example_name == "crowding":
            example_crowding_analysis()
        elif example_name == "qc":
            example_qc_compilation()
        elif example_name == "all":
            run_all_examples()
        else:
            print(f"Unknown example: {example_name}")
            print("Available: basic, dose-response, genotype, crowding, qc, all")
    else:
        run_all_examples()
