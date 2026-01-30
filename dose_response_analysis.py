"""
Dose-Response Analysis Utilities for DNA Damage Pipeline
=========================================================

This module provides:
  - Dose-response curve fitting (Hill equation)
  - EC50/IC50 estimation
  - Statistical comparisons between genotypes
  - Plotting utilities for dose-response visualization

Integrates with the main DNADamageAnalysisPipeline.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from dataclasses import dataclass


# =============================================================================
# DOSE-RESPONSE MODELS
# =============================================================================

def hill_equation(
    x: np.ndarray,
    bottom: float,
    top: float,
    ec50: float,
    hill_slope: float,
) -> np.ndarray:
    """
    Four-parameter Hill equation for dose-response.
    
    y = bottom + (top - bottom) / (1 + (ec50/x)^hill_slope)
    
    Parameters
    ----------
    x : array
        Dose values (concentration)
    bottom : float
        Minimum response (at infinite dose)
    top : float
        Maximum response (at zero dose)
    ec50 : float
        Dose at 50% response
    hill_slope : float
        Hill coefficient (steepness)
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        return bottom + (top - bottom) / (1 + (ec50 / x) ** hill_slope)


def inverse_hill(
    y: float,
    bottom: float,
    top: float,
    ec50: float,
    hill_slope: float,
) -> float:
    """Calculate dose for a given response level."""
    if y <= bottom or y >= top:
        return np.nan
    
    ratio = (top - bottom) / (y - bottom) - 1
    if ratio <= 0:
        return np.nan
    
    return ec50 * (ratio ** (1 / hill_slope))


@dataclass
class DoseResponseFit:
    """Results from dose-response curve fitting."""
    bottom: float
    top: float
    ec50: float
    hill_slope: float
    r_squared: float
    se_ec50: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    converged: bool = True
    doses: Optional[np.ndarray] = None
    responses: Optional[np.ndarray] = None
    fitted_responses: Optional[np.ndarray] = None


def fit_dose_response(
    doses: np.ndarray,
    responses: np.ndarray,
    *,
    initial_guess: Optional[Tuple[float, float, float, float]] = None,
    max_iter: int = 5000,
    bounds: Optional[Tuple[Tuple, Tuple]] = None,
) -> DoseResponseFit:
    """
    Fit a four-parameter Hill equation to dose-response data.
    
    Parameters
    ----------
    doses : array
        Dose values (concentrations), must be positive
    responses : array
        Response values
    initial_guess : tuple, optional
        Initial (bottom, top, ec50, hill_slope)
    max_iter : int
        Maximum iterations for optimization
    bounds : tuple, optional
        Bounds for parameters ((lower,), (upper,))
    
    Returns
    -------
    DoseResponseFit
        Fitted parameters and statistics
    """
    doses = np.asarray(doses, dtype=np.float64)
    responses = np.asarray(responses, dtype=np.float64)
    
    # Remove NaN and non-positive doses
    mask = ~np.isnan(doses) & ~np.isnan(responses) & (doses > 0)
    doses = doses[mask]
    responses = responses[mask]
    
    if len(doses) < 4:
        return DoseResponseFit(
            bottom=np.nan, top=np.nan, ec50=np.nan, hill_slope=np.nan,
            r_squared=np.nan, converged=False,
            doses=doses, responses=responses
        )
    
    # Initial guess
    if initial_guess is None:
        bottom_init = np.min(responses)
        top_init = np.max(responses)
        ec50_init = np.median(doses)
        hill_init = 1.0
        initial_guess = (bottom_init, top_init, ec50_init, hill_init)
    
    # Bounds
    if bounds is None:
        bounds = (
            (0, 0, 1e-12, 0.1),           # lower bounds
            (np.inf, np.inf, np.inf, 10)   # upper bounds
        )
    
    try:
        popt, pcov = curve_fit(
            hill_equation,
            doses,
            responses,
            p0=initial_guess,
            bounds=bounds,
            maxfev=max_iter,
        )
        
        bottom, top, ec50, hill_slope = popt
        
        # Calculate R-squared
        fitted = hill_equation(doses, *popt)
        ss_res = np.sum((responses - fitted) ** 2)
        ss_tot = np.sum((responses - np.mean(responses)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Standard error of EC50
        if pcov is not None and not np.any(np.isinf(pcov)):
            se_ec50 = np.sqrt(pcov[2, 2])
            # 95% CI
            ci_lower = ec50 - 1.96 * se_ec50
            ci_upper = ec50 + 1.96 * se_ec50
        else:
            se_ec50 = None
            ci_lower = None
            ci_upper = None
        
        return DoseResponseFit(
            bottom=bottom,
            top=top,
            ec50=ec50,
            hill_slope=hill_slope,
            r_squared=r_squared,
            se_ec50=se_ec50,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            converged=True,
            doses=doses,
            responses=responses,
            fitted_responses=fitted,
        )
        
    except (RuntimeError, ValueError) as e:
        warnings.warn(f"Dose-response fit failed: {e}")
        return DoseResponseFit(
            bottom=np.nan, top=np.nan, ec50=np.nan, hill_slope=np.nan,
            r_squared=np.nan, converged=False,
            doses=doses, responses=responses
        )


# =============================================================================
# DOSE-RESPONSE ANALYSIS
# =============================================================================

class DoseResponseAnalyzer:
    """
    Analyze dose-response relationships from DNA damage data.
    """
    
    def __init__(
        self,
        dose_column: str = "dilut_um",
        control_label: str = "DMSO",
    ):
        self.dose_column = dose_column
        self.control_label = control_label
    
    def fit_drug_response(
        self,
        df: pd.DataFrame,
        response_column: str,
        groupby: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Fit dose-response curves for each drug/genotype combination.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data with dose and response columns
        response_column : str
            Column to use as response variable
        groupby : list, optional
            Columns to group by (default: ['genotype', 'drug'])
        
        Returns
        -------
        pd.DataFrame
            Fit results per group
        """
        if groupby is None:
            groupby = ['genotype', 'drug']
        groupby = [c for c in groupby if c in df.columns]
        
        if not groupby:
            groupby = ['drug'] if 'drug' in df.columns else []
        
        results = []
        
        if groupby:
            for name, group in df.groupby(groupby):
                if isinstance(name, str):
                    name = (name,)
                
                # Get doses and responses
                doses = group[self.dose_column].values
                responses = group[response_column].values
                
                # Fit curve
                fit = fit_dose_response(doses, responses)
                
                # Build result row
                row = dict(zip(groupby, name))
                row.update({
                    'response_column': response_column,
                    'n_points': len(doses),
                    'ec50': fit.ec50,
                    'hill_slope': fit.hill_slope,
                    'bottom': fit.bottom,
                    'top': fit.top,
                    'r_squared': fit.r_squared,
                    'se_ec50': fit.se_ec50,
                    'ec50_ci_lower': fit.ci_lower,
                    'ec50_ci_upper': fit.ci_upper,
                    'converged': fit.converged,
                })
                results.append(row)
        else:
            doses = df[self.dose_column].values
            responses = df[response_column].values
            fit = fit_dose_response(doses, responses)
            
            results.append({
                'response_column': response_column,
                'n_points': len(doses),
                'ec50': fit.ec50,
                'hill_slope': fit.hill_slope,
                'bottom': fit.bottom,
                'top': fit.top,
                'r_squared': fit.r_squared,
                'se_ec50': fit.se_ec50,
                'ec50_ci_lower': fit.ci_lower,
                'ec50_ci_upper': fit.ci_upper,
                'converged': fit.converged,
            })
        
        return pd.DataFrame(results)
    
    def compute_response_at_dose(
        self,
        df: pd.DataFrame,
        response_columns: List[str],
        target_dose: float,
        groupby: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compute mean response at a specific dose for comparison.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data with dose and response columns
        response_columns : list
            Columns to analyze
        target_dose : float
            Target dose (will use nearest available)
        groupby : list, optional
            Columns to group by
        
        Returns
        -------
        pd.DataFrame
            Response statistics at target dose
        """
        if groupby is None:
            groupby = ['genotype', 'drug']
        groupby = [c for c in groupby if c in df.columns]
        
        # Find closest dose to target
        doses = df[self.dose_column].dropna().unique()
        if len(doses) == 0:
            return pd.DataFrame()
        
        closest_dose = doses[np.argmin(np.abs(doses - target_dose))]
        
        # Filter to closest dose
        dose_df = df[np.isclose(df[self.dose_column], closest_dose, rtol=0.1)]
        
        results = []
        
        if groupby:
            for name, group in dose_df.groupby(groupby):
                if isinstance(name, str):
                    name = (name,)
                
                row = dict(zip(groupby, name))
                row['target_dose'] = target_dose
                row['actual_dose'] = closest_dose
                row['n_samples'] = len(group)
                
                for col in response_columns:
                    if col in group.columns:
                        row[f'{col}_mean'] = group[col].mean()
                        row[f'{col}_std'] = group[col].std()
                        row[f'{col}_sem'] = group[col].sem()
                
                results.append(row)
        else:
            row = {
                'target_dose': target_dose,
                'actual_dose': closest_dose,
                'n_samples': len(dose_df),
            }
            for col in response_columns:
                if col in dose_df.columns:
                    row[f'{col}_mean'] = dose_df[col].mean()
                    row[f'{col}_std'] = dose_df[col].std()
                    row[f'{col}_sem'] = dose_df[col].sem()
            results.append(row)
        
        return pd.DataFrame(results)


# =============================================================================
# STATISTICAL COMPARISONS
# =============================================================================

class StatisticalComparator:
    """
    Statistical comparisons between genotypes and treatments.
    """
    
    def compare_ec50s(
        self,
        fit_results: pd.DataFrame,
        groupby: str = 'genotype',
        drug_col: str = 'drug',
    ) -> pd.DataFrame:
        """
        Compare EC50 values between groups using t-test approximation.
        
        Parameters
        ----------
        fit_results : pd.DataFrame
            Results from DoseResponseAnalyzer.fit_drug_response
        groupby : str
            Column to compare (e.g., 'genotype')
        drug_col : str
            Column identifying drugs
        
        Returns
        -------
        pd.DataFrame
            Pairwise comparison results
        """
        results = []
        
        if drug_col not in fit_results.columns:
            return pd.DataFrame()
        
        for drug, drug_df in fit_results.groupby(drug_col):
            groups = drug_df[groupby].unique()
            
            if len(groups) < 2:
                continue
            
            # Pairwise comparisons
            for i, g1 in enumerate(groups):
                for g2 in groups[i+1:]:
                    df1 = drug_df[drug_df[groupby] == g1]
                    df2 = drug_df[drug_df[groupby] == g2]
                    
                    if len(df1) == 0 or len(df2) == 0:
                        continue
                    
                    ec50_1 = df1['ec50'].values[0]
                    ec50_2 = df2['ec50'].values[0]
                    se1 = df1['se_ec50'].values[0] if 'se_ec50' in df1.columns else None
                    se2 = df2['se_ec50'].values[0] if 'se_ec50' in df2.columns else None
                    
                    # Log-ratio of EC50s
                    if ec50_1 > 0 and ec50_2 > 0:
                        ratio = ec50_1 / ec50_2
                        log_ratio = np.log10(ratio)
                    else:
                        ratio = np.nan
                        log_ratio = np.nan
                    
                    # Approximate z-test if SEs available
                    if se1 is not None and se2 is not None and se1 > 0 and se2 > 0:
                        se_diff = np.sqrt(se1**2 + se2**2)
                        z_stat = (ec50_1 - ec50_2) / se_diff
                        p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))
                    else:
                        z_stat = np.nan
                        p_value = np.nan
                    
                    results.append({
                        drug_col: drug,
                        f'{groupby}_1': g1,
                        f'{groupby}_2': g2,
                        'ec50_1': ec50_1,
                        'ec50_2': ec50_2,
                        'ec50_ratio': ratio,
                        'log10_ec50_ratio': log_ratio,
                        'z_statistic': z_stat,
                        'p_value': p_value,
                    })
        
        result_df = pd.DataFrame(results)
        
        # Multiple testing correction (Bonferroni)
        if 'p_value' in result_df.columns and len(result_df) > 0:
            n_tests = result_df['p_value'].notna().sum()
            if n_tests > 0:
                result_df['p_adjusted'] = result_df['p_value'] * n_tests
                result_df['p_adjusted'] = result_df['p_adjusted'].clip(upper=1.0)
        
        return result_df
    
    def compare_responses_at_dose(
        self,
        df: pd.DataFrame,
        response_col: str,
        dose_col: str = 'dilut_um',
        groupby: str = 'genotype',
        drug_col: str = 'drug',
    ) -> pd.DataFrame:
        """
        Compare response values between groups at each dose using Mann-Whitney U.
        
        Parameters
        ----------
        df : pd.DataFrame
            Single-cell or well-level data
        response_col : str
            Response column to compare
        dose_col : str
            Dose column
        groupby : str
            Column to compare (e.g., 'genotype')
        drug_col : str
            Column identifying drugs
        
        Returns
        -------
        pd.DataFrame
            Statistical comparison results per dose/drug
        """
        results = []
        
        for (drug, dose), dose_df in df.groupby([drug_col, dose_col]):
            groups = dose_df[groupby].unique()
            
            if len(groups) < 2:
                continue
            
            for i, g1 in enumerate(groups):
                for g2 in groups[i+1:]:
                    vals1 = dose_df[dose_df[groupby] == g1][response_col].dropna()
                    vals2 = dose_df[dose_df[groupby] == g2][response_col].dropna()
                    
                    if len(vals1) < 3 or len(vals2) < 3:
                        continue
                    
                    # Mann-Whitney U test
                    try:
                        stat, p_value = stats.mannwhitneyu(
                            vals1, vals2, alternative='two-sided'
                        )
                    except ValueError:
                        stat, p_value = np.nan, np.nan
                    
                    # Effect size (rank-biserial correlation)
                    n1, n2 = len(vals1), len(vals2)
                    if n1 > 0 and n2 > 0:
                        r = 1 - (2 * stat) / (n1 * n2)
                    else:
                        r = np.nan
                    
                    results.append({
                        drug_col: drug,
                        dose_col: dose,
                        f'{groupby}_1': g1,
                        f'{groupby}_2': g2,
                        'n_1': n1,
                        'n_2': n2,
                        'mean_1': vals1.mean(),
                        'mean_2': vals2.mean(),
                        'median_1': vals1.median(),
                        'median_2': vals2.median(),
                        'u_statistic': stat,
                        'p_value': p_value,
                        'effect_size_r': r,
                    })
        
        result_df = pd.DataFrame(results)
        
        # FDR correction
        if 'p_value' in result_df.columns and len(result_df) > 0:
            p_vals = result_df['p_value'].values
            mask = ~np.isnan(p_vals)
            if mask.sum() > 0:
                _, p_adj, _, _ = stats.false_discovery_control(
                    p_vals[mask], method='bh'
                ) if hasattr(stats, 'false_discovery_control') else (
                    None, p_vals[mask] * mask.sum(), None, None
                )
                result_df.loc[mask, 'p_adjusted'] = p_adj
        
        return result_df


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

class ResponseSummarizer:
    """
    Compute summary statistics for dose-response data.
    """
    
    def __init__(
        self,
        dose_col: str = 'dilut_um',
        control_label: str = 'DMSO',
    ):
        self.dose_col = dose_col
        self.control_label = control_label
    
    def summarize_by_dose(
        self,
        df: pd.DataFrame,
        response_cols: List[str],
        groupby: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compute summary statistics per dose level.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to summarize
        response_cols : list
            Response columns to summarize
        groupby : list, optional
            Additional grouping columns
        
        Returns
        -------
        pd.DataFrame
            Summary statistics per dose
        """
        if groupby is None:
            groupby = ['genotype', 'drug']
        
        groupby = [c for c in groupby if c in df.columns]
        all_groups = groupby + [self.dose_col]
        all_groups = [c for c in all_groups if c in df.columns]
        
        if not all_groups:
            return pd.DataFrame()
        
        # Compute statistics
        agg_dict = {}
        for col in response_cols:
            if col in df.columns:
                agg_dict[col] = ['count', 'mean', 'std', 'sem', 'median',
                                 lambda x: x.astype(float).quantile(0.25),
                                 lambda x: x.astype(float).quantile(0.75)]
        
        if not agg_dict:
            return pd.DataFrame()
        
        summary = df.groupby(all_groups).agg(agg_dict)
        
        # Flatten column names
        new_cols = []
        for col, stat in summary.columns:
            if callable(stat):
                stat_name = 'q25' if '0.25' in str(stat) else 'q75'
            else:
                stat_name = stat
            new_cols.append(f"{col}_{stat_name}")
        summary.columns = new_cols
        
        return summary.reset_index()
    
    def compute_fold_change_vs_control(
        self,
        df: pd.DataFrame,
        response_cols: List[str],
        groupby: Optional[List[str]] = None,
        control_col: str = 'is_control',
    ) -> pd.DataFrame:
        """
        Compute fold-change relative to DMSO control.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data with control and treatment samples
        response_cols : list
            Response columns
        groupby : list, optional
            Grouping columns (default: genotype, drug)
        control_col : str
            Column indicating control samples
        
        Returns
        -------
        pd.DataFrame
            Fold-change values per dose
        """
        if groupby is None:
            groupby = ['genotype', 'drug']
        groupby = [c for c in groupby if c in df.columns]
        
        if control_col not in df.columns:
            # Try to identify controls from dose column
            if self.dose_col in df.columns:
                df = df.copy()
                df[control_col] = df[self.dose_col] == 0
            else:
                return pd.DataFrame()
        
        results = []
        
        for name, group in df.groupby(groupby) if groupby else [(None, df)]:
            if isinstance(name, str):
                name = (name,)
            
            # Get control mean
            controls = group[group[control_col]]
            if len(controls) == 0:
                continue
            
            control_means = {}
            for col in response_cols:
                if col in controls.columns:
                    control_means[col] = controls[col].mean()
            
            # Compute fold-change for each dose
            for dose, dose_group in group.groupby(self.dose_col):
                row = dict(zip(groupby, name)) if groupby else {}
                row[self.dose_col] = dose
                row['n_samples'] = len(dose_group)
                row['is_control'] = dose_group[control_col].all() if control_col in dose_group.columns else False
                
                for col in response_cols:
                    if col in dose_group.columns and col in control_means:
                        mean_val = dose_group[col].mean()
                        ctrl_val = control_means[col]
                        
                        row[f'{col}_mean'] = mean_val
                        row[f'{col}_control_mean'] = ctrl_val
                        
                        if ctrl_val != 0:
                            row[f'{col}_fold_change'] = mean_val / ctrl_val
                            row[f'{col}_log2_fc'] = np.log2(mean_val / ctrl_val) if mean_val > 0 else np.nan
                        else:
                            row[f'{col}_fold_change'] = np.nan
                            row[f'{col}_log2_fc'] = np.nan
                
                results.append(row)
        
        return pd.DataFrame(results)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analyze_dose_response(
    df: pd.DataFrame,
    response_columns: List[str],
    dose_column: str = 'dilut_um',
    groupby: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Comprehensive dose-response analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with dose and response columns
    response_columns : list
        Response columns to analyze
    dose_column : str
        Dose column name
    groupby : list, optional
        Grouping columns
    
    Returns
    -------
    dict
        Dictionary with analysis results:
        - 'fits': Curve fit parameters
        - 'summary': Summary statistics by dose
        - 'fold_change': Fold-change vs control
    """
    analyzer = DoseResponseAnalyzer(dose_column=dose_column)
    summarizer = ResponseSummarizer(dose_col=dose_column)
    
    results = {}
    
    # Fit dose-response curves
    fit_results = []
    for col in response_columns:
        if col in df.columns:
            fits = analyzer.fit_drug_response(df, col, groupby=groupby)
            fit_results.append(fits)
    
    if fit_results:
        results['fits'] = pd.concat(fit_results, ignore_index=True)
    
    # Summary statistics
    results['summary'] = summarizer.summarize_by_dose(df, response_columns, groupby)
    
    # Fold-change vs control
    results['fold_change'] = summarizer.compute_fold_change_vs_control(
        df, response_columns, groupby
    )
    
    return results
