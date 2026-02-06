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


def hill_equation_log10(
    x_log10: np.ndarray,
    bottom: float,
    top: float,
    log10_ec50: float,
    hill_slope: float,
) -> np.ndarray:
    """Hill equation parameterized on log10 dose."""
    with np.errstate(over='ignore', invalid='ignore'):
        exponent = (log10_ec50 - x_log10) * hill_slope
        return bottom + (top - bottom) / (1 + np.power(10.0, exponent))


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
    log10_ec50: float
    hill_slope: float
    r_squared: float
    rmse: float
    aic: float
    model: str
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
    weights: Optional[np.ndarray] = None,
    control_response: Optional[float] = None,
    initial_guess: Optional[Tuple[float, float, float, float]] = None,
    max_iter: int = 5000,
    bounds: Optional[Tuple[Tuple, Tuple]] = None,
    model_candidates: Optional[Sequence[str]] = None,
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
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
    
    # Remove NaN and non-positive doses
    mask = ~np.isnan(doses) & ~np.isnan(responses) & (doses > 0)
    doses = doses[mask]
    responses = responses[mask]
    if weights is not None:
        weights = weights[mask]
    
    if len(doses) < 4:
        return DoseResponseFit(
            bottom=np.nan, top=np.nan, ec50=np.nan, log10_ec50=np.nan,
            hill_slope=np.nan, r_squared=np.nan, rmse=np.nan, aic=np.nan,
            model="none", converged=False,
            doses=doses, responses=responses
        )

    if model_candidates is None:
        model_candidates = ("4p", "3p_top")

    x_log10 = np.log10(doses)

    def _default_initials() -> List[Tuple[float, float, float, float]]:
        bottom_init = np.nanmin(responses)
        top_init = np.nanmax(responses) if control_response is None else control_response
        log_ec50_init = np.nanmedian(x_log10)
        hill_init = 1.0
        return [
            (bottom_init, top_init, log_ec50_init, hill_init),
            (bottom_init, top_init, log_ec50_init, 0.5),
            (bottom_init, top_init, log_ec50_init, 2.0),
        ]

    initial_guesses = [initial_guess] if initial_guess else _default_initials()

    if bounds is None:
        resp_min = np.nanmin(responses)
        resp_max = np.nanmax(responses)
        bounds = (
            (resp_min - abs(resp_min), resp_min - abs(resp_min), -12, 0.1),
            (resp_max + abs(resp_max), resp_max + abs(resp_max), 12, 10),
        )

    best_fit: Optional[DoseResponseFit] = None

    for model_type in model_candidates:
        top_fixed = None
        if model_type == "3p_top":
            top_fixed = control_response if control_response is not None else np.nanmax(responses)

        for guess in initial_guesses:
            try:
                if model_type == "4p":
                    p0 = guess
                    popt, pcov = curve_fit(
                        hill_equation_log10,
                        x_log10,
                        responses,
                        p0=p0,
                        bounds=bounds,
                        maxfev=max_iter,
                        sigma=_weights_to_sigma(weights),
                        absolute_sigma=False,
                    )
                    bottom, top, log_ec50, hill_slope = popt
                elif model_type == "3p_top":
                    if top_fixed is None:
                        continue
                    p0 = (guess[0], guess[2], guess[3])
                    popt, pcov = curve_fit(
                        lambda x, bottom, log_ec50, hill_slope: hill_equation_log10(
                            x, bottom, top_fixed, log_ec50, hill_slope
                        ),
                        x_log10,
                        responses,
                        p0=p0,
                        bounds=(
                            (bounds[0][0], bounds[0][2], bounds[0][3]),
                            (bounds[1][0], bounds[1][2], bounds[1][3]),
                        ),
                        maxfev=max_iter,
                        sigma=_weights_to_sigma(weights),
                        absolute_sigma=False,
                    )
                    bottom, log_ec50, hill_slope = popt
                    top = top_fixed
                else:
                    continue

                fitted = hill_equation_log10(x_log10, bottom, top, log_ec50, hill_slope)
                ss_res = np.sum((responses - fitted) ** 2)
                ss_tot = np.sum((responses - np.mean(responses)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                rmse = np.sqrt(ss_res / len(responses))
                k_params = 4 if model_type == "4p" else 3
                aic = _aic(ss_res, len(responses), k_params)

                log_ec50_index = 2 if model_type == "4p" else 1
                se_ec50, ci_lower, ci_upper = _compute_ec50_ci(pcov, log_ec50, log_ec50_index)

                fit = DoseResponseFit(
                    bottom=bottom,
                    top=top,
                    ec50=10 ** log_ec50,
                    log10_ec50=log_ec50,
                    hill_slope=hill_slope,
                    r_squared=r_squared,
                    rmse=rmse,
                    aic=aic,
                    model=model_type,
                    se_ec50=se_ec50,
                    ci_lower=ci_lower,
                    ci_upper=ci_upper,
                    converged=True,
                    doses=doses,
                    responses=responses,
                    fitted_responses=fitted,
                )

                if best_fit is None or fit.aic < best_fit.aic:
                    best_fit = fit
            except (RuntimeError, ValueError):
                continue

    if best_fit is None:
        warnings.warn("Dose-response fit failed")
        return DoseResponseFit(
            bottom=np.nan, top=np.nan, ec50=np.nan, log10_ec50=np.nan,
            hill_slope=np.nan, r_squared=np.nan, rmse=np.nan, aic=np.nan,
            model="none", converged=False,
            doses=doses, responses=responses
        )

    return best_fit


def _weights_to_sigma(weights: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if weights is None:
        return None
    safe_weights = np.where(weights <= 0, np.nan, weights)
    sigma = 1.0 / np.sqrt(safe_weights)
    return np.where(np.isfinite(sigma), sigma, 1.0)


def _compute_ec50_ci(
    pcov: Optional[np.ndarray],
    log_ec50: float,
    log_ec50_index: int,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if pcov is None or np.any(np.isinf(pcov)):
        return None, None, None
    if pcov.shape[0] <= log_ec50_index:
        return None, None, None
    log_se = np.sqrt(pcov[log_ec50_index, log_ec50_index])
    if not np.isfinite(log_se):
        return None, None, None
    ec50 = 10 ** log_ec50
    se_ec50 = ec50 * np.log(10) * log_se
    ci_lower = 10 ** (log_ec50 - 1.96 * log_se)
    ci_upper = 10 ** (log_ec50 + 1.96 * log_se)
    return se_ec50, ci_lower, ci_upper


def _aic(ss_res: float, n: int, k: int) -> float:
    if n <= 0 or ss_res <= 0:
        return np.nan
    return n * np.log(ss_res / n) + 2 * k


def _bootstrap_resample(df: pd.DataFrame, strata_cols: List[str], rng: np.random.Generator) -> pd.DataFrame:
    if not strata_cols:
        return df.sample(len(df), replace=True, random_state=rng.integers(0, 1e9))
    samples = []
    for _, group in df.groupby(strata_cols):
        if group.empty:
            continue
        sampled = group.sample(len(group), replace=True, random_state=rng.integers(0, 1e9))
        samples.append(sampled)
    return pd.concat(samples, ignore_index=True) if samples else df.copy()


def _control_mean(df: pd.DataFrame, response_col: str) -> Optional[float]:
    if 'is_control' in df.columns:
        control_rows = df[df['is_control']]
        if not control_rows.empty:
            return control_rows[response_col].mean()
    return None


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
        weight_column: Optional[str] = None,
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
                weights = group[weight_column].values if weight_column and weight_column in group.columns else None
                control_mean = None
                if 'is_control' in group.columns:
                    control_rows = group[group['is_control']]
                    if not control_rows.empty:
                        control_mean = control_rows[response_column].mean()

                # Fit curve
                fit = fit_dose_response(
                    doses,
                    responses,
                    weights=weights,
                    control_response=control_mean,
                )
                
                # Build result row
                row = dict(zip(groupby, name))
                row.update({
                    'response_column': response_column,
                    'n_points': len(fit.doses) if fit.doses is not None else 0,
                    'ec50': fit.ec50,
                    'log10_ec50': fit.log10_ec50,
                    'hill_slope': fit.hill_slope,
                    'bottom': fit.bottom,
                    'top': fit.top,
                    'r_squared': fit.r_squared,
                    'rmse': fit.rmse,
                    'aic': fit.aic,
                    'model': fit.model,
                    'se_ec50': fit.se_ec50,
                    'ec50_ci_lower': fit.ci_lower,
                    'ec50_ci_upper': fit.ci_upper,
                    'converged': fit.converged,
                })
                results.append(row)
        else:
            doses = df[self.dose_column].values
            responses = df[response_column].values
            weights = df[weight_column].values if weight_column and weight_column in df.columns else None
            control_mean = None
            if 'is_control' in df.columns:
                control_rows = df[df['is_control']]
                if not control_rows.empty:
                    control_mean = control_rows[response_column].mean()
            fit = fit_dose_response(
                doses,
                responses,
                weights=weights,
                control_response=control_mean,
            )
            
            results.append({
                'response_column': response_column,
                'n_points': len(fit.doses) if fit.doses is not None else 0,
                'ec50': fit.ec50,
                'log10_ec50': fit.log10_ec50,
                'hill_slope': fit.hill_slope,
                'bottom': fit.bottom,
                'top': fit.top,
                'r_squared': fit.r_squared,
                'rmse': fit.rmse,
                'aic': fit.aic,
                'model': fit.model,
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

    def compare_ec50s_bootstrap(
        self,
        df: pd.DataFrame,
        response_col: str,
        dose_col: str = 'dilut_um',
        groupby: str = 'genotype',
        drug_col: str = 'drug',
        plate_col: str = 'plate',
        replicate_col: str = 'replicate',
        weight_col: Optional[str] = None,
        n_boot: int = 500,
        random_state: Optional[int] = 42,
    ) -> pd.DataFrame:
        """
        Compare EC50 values via stratified cluster bootstrap at the well level.
        """
        rng = np.random.default_rng(random_state)
        results = []

        strata_cols = [c for c in [plate_col, replicate_col] if c in df.columns]

        for drug, drug_df in df.groupby(drug_col):
            groups = drug_df[groupby].unique()
            if len(groups) < 2:
                continue

            for i, g1 in enumerate(groups):
                for g2 in groups[i + 1:]:
                    d1 = drug_df[drug_df[groupby] == g1]
                    d2 = drug_df[drug_df[groupby] == g2]
                    if d1.empty or d2.empty:
                        continue

                    deltas = []
                    for _ in range(n_boot):
                        s1 = _bootstrap_resample(d1, strata_cols, rng)
                        s2 = _bootstrap_resample(d2, strata_cols, rng)

                        fit1 = fit_dose_response(
                            s1[dose_col].values,
                            s1[response_col].values,
                            weights=s1[weight_col].values if weight_col and weight_col in s1.columns else None,
                            control_response=_control_mean(s1, response_col),
                        )
                        fit2 = fit_dose_response(
                            s2[dose_col].values,
                            s2[response_col].values,
                            weights=s2[weight_col].values if weight_col and weight_col in s2.columns else None,
                            control_response=_control_mean(s2, response_col),
                        )

                        if np.isfinite(fit1.log10_ec50) and np.isfinite(fit2.log10_ec50):
                            deltas.append(fit1.log10_ec50 - fit2.log10_ec50)

                    if not deltas:
                        continue

                    deltas = np.array(deltas)
                    ci_lower, ci_upper = np.percentile(deltas, [2.5, 97.5])
                    p_value = 2 * min(
                        np.mean(deltas <= 0),
                        np.mean(deltas >= 0),
                    )

                    results.append({
                        drug_col: drug,
                        f'{groupby}_1': g1,
                        f'{groupby}_2': g2,
                        'response_column': response_col,
                        'delta_log10_ec50': np.mean(deltas),
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'p_value': p_value,
                        'n_boot': len(deltas),
                        'n_wells_1': len(d1),
                        'n_wells_2': len(d2),
                    })

        result_df = pd.DataFrame(results)
        return result_df
    
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
        baseline_cols: Optional[List[str]] = None,
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
        response_cols = [c for c in response_cols if c in df.columns]
        if not response_cols:
            return pd.DataFrame()

        if groupby is None:
            groupby = ['genotype', 'drug']
        groupby = [c for c in groupby if c in df.columns]
        if baseline_cols is None:
            baseline_cols = ['plate']
        baseline_cols = [c for c in baseline_cols if c in df.columns]
        
        if control_col not in df.columns:
            # Try to identify controls from dose column
            if self.dose_col in df.columns:
                df = df.copy()
                df[control_col] = df[self.dose_col] == 0
            else:
                return pd.DataFrame()
        
        results = []

        control_group_cols = groupby + baseline_cols
        if not control_group_cols:
            control_group_cols = [self.dose_col] if self.dose_col in df.columns else []

        control_df = df[df[control_col]] if control_col in df.columns else pd.DataFrame()
        if control_df.empty:
            return pd.DataFrame()

        control_stats = (
            control_df.groupby(control_group_cols)[response_cols]
            .mean()
            .reset_index()
            .rename(columns={col: f"{col}_control_mean" for col in response_cols})
        )

        dose_group_cols = control_group_cols + [self.dose_col]
        summary = df.groupby(dose_group_cols)[response_cols].mean().reset_index()
        summary = summary.merge(control_stats, on=control_group_cols, how='left')

        for col in response_cols:
            ctrl_col = f"{col}_control_mean"
            if ctrl_col not in summary.columns:
                continue
            summary[f"{col}_fold_change"] = summary[col] / summary[ctrl_col]
            summary[f"{col}_log2_fc"] = np.where(
                (summary[col] > 0) & (summary[ctrl_col] > 0),
                np.log2(summary[col] / summary[ctrl_col]),
                np.nan,
            )

        return summary


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
