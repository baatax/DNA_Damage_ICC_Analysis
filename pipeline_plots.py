"""
Plotting Module for DNA Damage ICC Analysis Pipeline
=====================================================

Generates publication-quality plots including:
  - PCA scatter plots colored by genotype
  - Feature contribution bar charts for each principal component
  - Crowding correction feature exclusion plots
  - Dose-correlated feature PCA and line plots
  - Variance explained scree plots

Uses non-interactive Agg backend for cluster compatibility.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib not available; plotting disabled")


# Colorblind-friendly palette
PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]


def _color_map(labels):
    """Map unique labels to colors."""
    unique = sorted(set(labels))
    return {l: PALETTE[i % len(PALETTE)] for i, l in enumerate(unique)}


# =========================================================================
# PCA Plots
# =========================================================================

def plot_pca_scatter(
    profiles: pd.DataFrame,
    pc_x: str = 'PC1',
    pc_y: str = 'PC2',
    color_col: str = 'genotype',
    var_ratio: Optional[Dict[str, float]] = None,
    title: str = 'PCA',
    output_path: Optional[Union[str, Path]] = None,
) -> Optional[object]:
    """PCA scatter plot colored by a grouping column."""
    if not HAS_MATPLOTLIB:
        return None

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = _color_map(profiles[color_col].values)

    for label, grp in profiles.groupby(color_col):
        ax.scatter(
            grp[pc_x], grp[pc_y],
            c=cmap[label], label=label,
            alpha=0.7, edgecolors='white', linewidth=0.5, s=50,
        )

    xlab = pc_x
    ylab = pc_y
    if var_ratio:
        xlab = f"{pc_x} ({var_ratio.get(pc_x, 0)*100:.1f}%)"
        ylab = f"{pc_y} ({var_ratio.get(pc_y, 0)*100:.1f}%)"

    ax.set_xlabel(xlab, fontsize=12)
    ax.set_ylabel(ylab, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(title=color_col, fontsize=9, title_fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


def plot_pca_feature_contributions(
    loadings: pd.DataFrame,
    var_explained: pd.DataFrame,
    n_features: int = 15,
    n_pcs: int = 5,
    output_dir: Optional[Union[str, Path]] = None,
    label: str = '',
) -> List[object]:
    """Horizontal bar chart of top feature loadings for each PC."""
    if not HAS_MATPLOTLIB:
        return []

    figs = []
    n_pcs = min(n_pcs, len(loadings.columns))

    for i in range(n_pcs):
        pc = f'PC{i+1}'
        if pc not in loadings.columns:
            continue

        vals = loadings[pc]
        top_idx = vals.abs().nlargest(n_features).index
        top = vals.loc[top_idx].sort_values()

        var_row = var_explained.loc[var_explained['PC'] == pc, 'variance_ratio']
        var_str = f" ({var_row.values[0]*100:.1f}%)" if len(var_row) > 0 else ""

        fig, ax = plt.subplots(figsize=(8, max(4, n_features * 0.35)))
        colors = ['#d62728' if v < 0 else '#1f77b4' for v in top.values]
        ax.barh(range(len(top)), top.values, color=colors)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top.index, fontsize=9)
        ax.set_xlabel('Loading', fontsize=12)
        ax.set_title(f'{label} {pc}{var_str} — Feature Contributions', fontsize=13)
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()

        if output_dir:
            safe = label.replace(' ', '_').replace('/', '_')
            fig.savefig(
                str(Path(output_dir) / f"feature_contributions_{safe}_{pc}.png"),
                dpi=150, bbox_inches='tight',
            )
            plt.close(fig)
        figs.append(fig)
    return figs


def plot_pca_variance_explained(
    var_explained: pd.DataFrame,
    title: str = 'PCA Variance Explained',
    output_path: Optional[Union[str, Path]] = None,
) -> Optional[object]:
    """Scree plot: individual + cumulative variance explained."""
    if not HAS_MATPLOTLIB:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    pcs = var_explained['PC'].values
    indiv = var_explained['variance_ratio'].values * 100
    cumul = var_explained['cumulative_variance_ratio'].values * 100

    ax1.bar(range(len(pcs)), indiv, color='#1f77b4', edgecolor='white')
    ax1.set_xticks(range(len(pcs)))
    ax1.set_xticklabels(pcs, rotation=45)
    ax1.set_ylabel('Variance Explained (%)')
    ax1.set_title('Individual')
    ax1.grid(True, axis='y', alpha=0.3)

    ax2.plot(range(len(pcs)), cumul, 'o-', color='#ff7f0e', linewidth=2)
    ax2.axhline(y=80, color='gray', linestyle='--', alpha=0.5, label='80%')
    ax2.set_xticks(range(len(pcs)))
    ax2.set_xticklabels(pcs, rotation=45)
    ax2.set_ylabel('Cumulative Variance (%)')
    ax2.set_title('Cumulative')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


# =========================================================================
# Crowding Correction Plot
# =========================================================================

def plot_crowding_excluded_features(
    correlations: pd.Series,
    threshold: float,
    excluded_features: List[str],
    output_path: Optional[Union[str, Path]] = None,
) -> Optional[object]:
    """Bar chart showing correlation of each feature with the crowding metric.

    Excluded features (above threshold) are highlighted in red.
    """
    if not HAS_MATPLOTLIB:
        return None

    sorted_abs = correlations.abs().sort_values(ascending=True)
    n = len(sorted_abs)
    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.22)))

    colors = [
        '#d62728' if f in excluded_features else '#1f77b4'
        for f in sorted_abs.index
    ]
    ax.barh(range(n), sorted_abs.values, color=colors)
    ax.set_yticks(range(n))
    ax.set_yticklabels(sorted_abs.index, fontsize=7)
    ax.axvline(x=threshold, color='red', linestyle='--', linewidth=1.5,
               label=f'Threshold (|r|={threshold})')
    ax.set_xlabel('|Spearman r| with Crowding Metric', fontsize=12)
    ax.set_title(
        'Feature–Crowding Correlation\n(Red = Excluded from Corrected Analysis)',
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


# =========================================================================
# Dose-Correlated Feature Plots
# =========================================================================

def plot_dose_feature_lines(
    summary: pd.DataFrame,
    features: List[str],
    dose_col: str = 'dilut_um',
    group_col: str = 'genotype',
    title: str = 'Dose-Correlated Features vs Drug Dose',
    output_path: Optional[Union[str, Path]] = None,
) -> Optional[object]:
    """Combined line plot: one subplot per feature, one line per genotype."""
    if not HAS_MATPLOTLIB or summary.empty or not features:
        return None

    n = len(features)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    cmap = _color_map(summary[group_col].values)

    for ax, feat in zip(axes, features):
        mean_col = f'{feat}_mean'
        sem_col = f'{feat}_sem'
        if mean_col not in summary.columns:
            ax.set_ylabel(feat, fontsize=10)
            ax.text(0.5, 0.5, 'no data', transform=ax.transAxes, ha='center')
            continue

        for label, grp in summary.groupby(group_col):
            grp_s = grp.sort_values(dose_col)
            doses = grp_s[dose_col].values
            means = grp_s[mean_col].values
            ax.plot(doses, means, 'o-', color=cmap[label], label=label,
                    markersize=5, linewidth=1.5)
            if sem_col in grp_s.columns:
                sems = grp_s[sem_col].values
                ax.fill_between(doses, means - sems, means + sems,
                                color=cmap[label], alpha=0.15)

        ax.set_ylabel(feat, fontsize=10)
        ax.set_xscale('symlog', linthresh=0.001)
        ax.grid(True, alpha=0.3)
        if ax is axes[0]:
            ax.legend(fontsize=8, ncol=min(6, len(cmap)),
                      loc='upper left', bbox_to_anchor=(0, 1.35))

    axes[-1].set_xlabel(f'Dose ({dose_col})', fontsize=12)
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


def plot_dose_correlation_ranking(
    correlations: pd.Series,
    n_top: int = 20,
    title: str = 'Feature–Dose Correlation Ranking',
    output_path: Optional[Union[str, Path]] = None,
) -> Optional[object]:
    """Bar chart of features ranked by absolute Spearman r with dose."""
    if not HAS_MATPLOTLIB:
        return None

    top = correlations.abs().nlargest(n_top).sort_values(ascending=True)
    # Use signed values for color
    signed = correlations.loc[top.index]

    fig, ax = plt.subplots(figsize=(8, max(4, n_top * 0.35)))
    colors = ['#d62728' if v < 0 else '#1f77b4' for v in signed.values]
    ax.barh(range(len(top)), signed.abs().values, color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=9)
    ax.set_xlabel('|Spearman r| with Drug Dose', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig
