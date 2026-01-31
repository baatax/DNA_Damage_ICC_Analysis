"""
PCA Visualization Plots for DNA Damage Analysis Pipeline
=========================================================

Generates:
  - PCA scatter plots (PC1 vs PC2) colored by genotype, drug, and dose
  - Feature loading bar charts colored by source imaging channel
  - Feature loading 2D scatter (PC1 vs PC2 loadings) colored by channel
  - Scree plots of variance explained

Channel classification maps feature names to their source imaging channel
using keyword matching. Features that cannot be assigned to a channel
(morphology, crowding, spatial metrics) are left gray ("uncolored").
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logger = logging.getLogger("DNADamagePipeline")


# =============================================================================
# CHANNEL CLASSIFICATION
# =============================================================================

CHANNEL_COLORS = {
    'gamma-H2AX': '#E63946',   # Red - DNA damage marker
    'Ki67':       '#2A9D8F',   # Teal - Proliferation marker
    'DAPI':       '#457B9D',   # Blue - Nuclear stain
    'SYTO RNA':   '#E9C46A',   # Gold - RNA stain
}

UNASSIGNED_COLOR = '#999999'

# Keywords for substring matching (case-insensitive).
# Order matters within each channel: first match wins.
CHANNEL_KEYWORDS = {
    'gamma-H2AX': ['foci', 'gamma_h2ax', 'h2ax'],
    'Ki67':       ['ki67'],
    'DAPI':       ['dapi'],
    'SYTO RNA':   ['syto_rna', 'syto'],
}


def classify_feature_channel(feature_name: str) -> Optional[str]:
    """Classify a feature name to its source imaging channel.

    Returns the channel name string, or None if the feature cannot be
    assigned to a specific channel (morphology, crowding, spatial, etc.).
    """
    lower = feature_name.lower()
    for channel, keywords in CHANNEL_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                return channel
    return None


def get_feature_colors(
    feature_names: List[str],
) -> Tuple[List[str], Dict[str, Optional[str]]]:
    """Return per-feature colors and a feature-to-channel mapping.

    Parameters
    ----------
    feature_names : list of str
        Feature names (e.g. from PCA loadings index).

    Returns
    -------
    colors : list of str
        Hex colour for each feature.
    channel_map : dict
        ``{feature_name: channel_name_or_None}``.
    """
    colors = []
    channel_map = {}
    for feat in feature_names:
        channel = classify_feature_channel(feat)
        channel_map[feat] = channel
        colors.append(CHANNEL_COLORS[channel] if channel else UNASSIGNED_COLOR)
    return colors, channel_map


# =============================================================================
# PCA SCATTER PLOTS
# =============================================================================

def plot_pca_scatter_by_category(
    profiles: pd.DataFrame,
    color_col: str,
    variance_ratios: np.ndarray,
    output_path: Path,
    pc_x: str = 'PC1',
    pc_y: str = 'PC2',
    title: str = '',
    figsize: Tuple[float, float] = (8, 6),
):
    """PCA scatter coloured by a categorical column (genotype, drug, …)."""
    if not HAS_MATPLOTLIB:
        return
    if color_col not in profiles.columns:
        return
    if pc_x not in profiles.columns or pc_y not in profiles.columns:
        return

    fig, ax = plt.subplots(figsize=figsize)

    categories = sorted(profiles[color_col].dropna().unique(), key=str)
    cmap = plt.get_cmap('tab10' if len(categories) <= 10 else 'tab20')

    for i, cat in enumerate(categories):
        mask = profiles[color_col] == cat
        color = cmap(i % cmap.N)
        ax.scatter(
            profiles.loc[mask, pc_x],
            profiles.loc[mask, pc_y],
            label=str(cat),
            color=color,
            alpha=0.7,
            s=40,
            edgecolors='white',
            linewidths=0.5,
        )

    pc_x_idx = int(pc_x.replace('PC', '')) - 1
    pc_y_idx = int(pc_y.replace('PC', '')) - 1

    ax.set_xlabel(f'{pc_x} ({variance_ratios[pc_x_idx]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'{pc_y} ({variance_ratios[pc_y_idx]*100:.1f}%)', fontsize=12)
    ax.set_title(title or f'PCA coloured by {color_col}', fontsize=13)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9,
              framealpha=0.9)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_pca_scatter_by_dose(
    profiles: pd.DataFrame,
    dose_col: str,
    variance_ratios: np.ndarray,
    output_path: Path,
    pc_x: str = 'PC1',
    pc_y: str = 'PC2',
    title: str = '',
    figsize: Tuple[float, float] = (8, 6),
):
    """PCA scatter coloured by dose on a log-scaled continuous colourmap."""
    if not HAS_MATPLOTLIB:
        return
    if dose_col not in profiles.columns:
        return
    if pc_x not in profiles.columns or pc_y not in profiles.columns:
        return

    fig, ax = plt.subplots(figsize=figsize)

    doses = profiles[dose_col].values.copy().astype(float)
    valid = ~np.isnan(doses) & (doses >= 0)
    positive = valid & (doses > 0)
    control = valid & (doses == 0)

    # Controls as gray ×-markers
    if control.any():
        ax.scatter(
            profiles.loc[control, pc_x],
            profiles.loc[control, pc_y],
            color='gray', alpha=0.6, s=40,
            label='Control (0)', marker='x',
        )

    # Treated wells on log-scaled viridis
    if positive.any():
        log_doses = np.log10(doses[positive])
        sc = ax.scatter(
            profiles.loc[positive, pc_x],
            profiles.loc[positive, pc_y],
            c=log_doses, cmap='viridis', alpha=0.7, s=40,
            edgecolors='white', linewidths=0.5,
        )
        cbar = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('log\u2081\u2080(dose \u00b5M)', fontsize=10)

    pc_x_idx = int(pc_x.replace('PC', '')) - 1
    pc_y_idx = int(pc_y.replace('PC', '')) - 1

    ax.set_xlabel(f'{pc_x} ({variance_ratios[pc_x_idx]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'{pc_y} ({variance_ratios[pc_y_idx]*100:.1f}%)', fontsize=12)
    ax.set_title(title or 'PCA coloured by dose', fontsize=13)
    if control.any():
        ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# SCREE PLOT
# =============================================================================

def plot_scree(
    variance_df: pd.DataFrame,
    output_path: Path,
    title: str = 'PCA Variance Explained',
    figsize: Tuple[float, float] = (8, 5),
):
    """Bar + cumulative-line scree plot."""
    if not HAS_MATPLOTLIB:
        return

    fig, ax1 = plt.subplots(figsize=figsize)

    pcs = variance_df['PC'].values
    ratios = variance_df['variance_ratio'].values * 100
    cumulative = variance_df['cumulative_variance_ratio'].values * 100

    x = np.arange(len(pcs))
    bars = ax1.bar(x, ratios, color='#457B9D', alpha=0.8, edgecolor='white')
    ax1.set_xlabel('Principal Component', fontsize=12)
    ax1.set_ylabel('Variance Explained (%)', fontsize=12, color='#457B9D')
    ax1.set_xticks(x)
    ax1.set_xticklabels(pcs, fontsize=9)
    ax1.tick_params(axis='y', labelcolor='#457B9D')

    for bar, ratio in zip(bars, ratios):
        if ratio > 2:
            ax1.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{ratio:.1f}%', ha='center', va='bottom', fontsize=8,
            )

    ax2 = ax1.twinx()
    ax2.plot(x, cumulative, 'o-', color='#E63946', markersize=5, linewidth=1.5)
    ax2.set_ylabel('Cumulative Variance (%)', fontsize=12, color='#E63946')
    ax2.tick_params(axis='y', labelcolor='#E63946')
    ax2.set_ylim(0, 105)

    ax1.set_title(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# FEATURE LOADING PLOTS
# =============================================================================

def plot_feature_loadings_bar(
    loadings_df: pd.DataFrame,
    pc_col: str,
    output_path: Path,
    title: str = '',
    top_n: int = 20,
    fig_width: float = 8,
):
    """Horizontal bar chart of feature loadings for one PC, coloured by channel.

    Shows the *top_n* features ranked by absolute loading value.
    """
    if not HAS_MATPLOTLIB:
        return
    if pc_col not in loadings_df.columns:
        return

    loadings = loadings_df[pc_col].copy()
    sorted_idx = loadings.abs().sort_values(ascending=False).index
    n_show = min(top_n, len(sorted_idx))
    top_features = sorted_idx[:n_show]

    values = loadings.loc[top_features].values
    feature_names = list(top_features)
    colors, channel_map = get_feature_colors(feature_names)

    fig_height = max(4, 0.35 * n_show)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    y = np.arange(n_show)
    ax.barh(y, values, color=colors, edgecolor='white', linewidth=0.5,
            alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(feature_names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(f'{pc_col} Loading', fontsize=12)
    ax.set_title(title or f'Feature Loadings on {pc_col}', fontsize=13)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.grid(True, axis='x', alpha=0.2)

    # Legend for channels present in shown features
    legend_elements = []
    channels_present = sorted(
        {ch for ch in channel_map.values() if ch is not None}
    )
    for ch in channels_present:
        legend_elements.append(
            Line2D([0], [0], marker='s', color='w',
                   markerfacecolor=CHANNEL_COLORS[ch],
                   markersize=10, label=ch)
        )
    if None in channel_map.values():
        legend_elements.append(
            Line2D([0], [0], marker='s', color='w',
                   markerfacecolor=UNASSIGNED_COLOR,
                   markersize=10, label='Other')
        )
    if legend_elements:
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
                  framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_feature_loadings_scatter(
    loadings_df: pd.DataFrame,
    output_path: Path,
    pc_x: str = 'PC1',
    pc_y: str = 'PC2',
    title: str = '',
    label_top_n: int = 10,
    figsize: Tuple[float, float] = (9, 8),
):
    """2-D scatter of feature loadings (PC1 vs PC2), coloured by channel.

    The *label_top_n* features with the largest combined loading magnitude
    are annotated with their names.
    """
    if not HAS_MATPLOTLIB:
        return
    if pc_x not in loadings_df.columns or pc_y not in loadings_df.columns:
        return

    feature_names = list(loadings_df.index)
    x_vals = loadings_df[pc_x].values
    y_vals = loadings_df[pc_y].values
    _colors, channel_map = get_feature_colors(feature_names)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot each channel group separately so the legend works
    channels_present = sorted(
        set(channel_map.values()),
        key=lambda c: (c is None, str(c)),
    )

    for channel in channels_present:
        mask = np.array([channel_map[f] == channel for f in feature_names])
        label = channel if channel is not None else 'Other'
        color = CHANNEL_COLORS.get(channel, UNASSIGNED_COLOR)  # type: ignore[arg-type]
        ax.scatter(
            x_vals[mask], y_vals[mask],
            color=color, alpha=0.7, s=50,
            edgecolors='white', linewidths=0.5, label=label,
        )

    # Annotate top features by combined magnitude
    magnitudes = np.sqrt(x_vals ** 2 + y_vals ** 2)
    n_label = min(label_top_n, len(feature_names))
    top_idx = np.argsort(magnitudes)[-n_label:]

    for idx in top_idx:
        ax.annotate(
            feature_names[idx],
            (x_vals[idx], y_vals[idx]),
            fontsize=7, alpha=0.8,
            xytext=(5, 5), textcoords='offset points',
        )

    ax.set_xlabel(f'{pc_x} Loading', fontsize=12)
    ax.set_ylabel(f'{pc_y} Loading', fontsize=12)
    ax.set_title(title or f'Feature Loadings: {pc_x} vs {pc_y}', fontsize=13)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=9, framealpha=0.9)

    # Reference circle at maximum loading magnitude
    if len(magnitudes) > 0:
        max_mag = np.max(magnitudes)
        if max_mag > 0:
            circle = plt.Circle(
                (0, 0), max_mag, fill=False,
                color='gray', linestyle=':', linewidth=0.5, alpha=0.3,
            )
            ax.add_patch(circle)

    ax.set_aspect('equal', adjustable='datalim')

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def generate_pca_plots(
    profiles_df: pd.DataFrame,
    loadings_df: pd.DataFrame,
    variance_df: pd.DataFrame,
    output_dir: Path,
    mode: str,
    experiment_name: str = '',
) -> Dict[str, Path]:
    """Generate all PCA visualisation plots for one normalisation mode.

    Parameters
    ----------
    profiles_df : pd.DataFrame
        Well-level profiles with PC columns and metadata
        (genotype, drug, dilut_um, …).
    loadings_df : pd.DataFrame
        Feature loadings with features as the index and PC columns.
    variance_df : pd.DataFrame
        Variance explained per PC (columns: PC, variance_ratio,
        cumulative_variance_ratio).
    output_dir : Path
        Directory in which to save plots.
    mode : str
        Normalisation mode label (``'per_genotype'`` or ``'across_all'``).
    experiment_name : str, optional
        Experiment name used in plot titles.

    Returns
    -------
    dict
        ``{plot_name: Path}`` for every plot that was successfully saved.
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available — skipping PCA plots")
        return {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files: Dict[str, Path] = {}

    variance_ratios = variance_df['variance_ratio'].values
    mode_label = mode.replace('_', '-')
    tp = f'{experiment_name} ' if experiment_name else ''

    # ── PCA scatter plots ──────────────────────────────────────────────

    if 'genotype' in profiles_df.columns:
        path = output_dir / f'pca_scatter_genotype_{mode}.png'
        plot_pca_scatter_by_category(
            profiles_df, 'genotype', variance_ratios, path,
            title=f'{tp}PCA by Genotype ({mode_label})',
        )
        output_files[f'pca_scatter_genotype_{mode}'] = path

    if 'drug' in profiles_df.columns:
        path = output_dir / f'pca_scatter_drug_{mode}.png'
        plot_pca_scatter_by_category(
            profiles_df, 'drug', variance_ratios, path,
            title=f'{tp}PCA by Drug ({mode_label})',
        )
        output_files[f'pca_scatter_drug_{mode}'] = path

    if 'dilut_um' in profiles_df.columns:
        path = output_dir / f'pca_scatter_dose_{mode}.png'
        plot_pca_scatter_by_dose(
            profiles_df, 'dilut_um', variance_ratios, path,
            title=f'{tp}PCA by Dose ({mode_label})',
        )
        output_files[f'pca_scatter_dose_{mode}'] = path

    # ── Scree plot ─────────────────────────────────────────────────────

    path = output_dir / f'pca_scree_{mode}.png'
    plot_scree(
        variance_df, path,
        title=f'{tp}Variance Explained ({mode_label})',
    )
    output_files[f'pca_scree_{mode}'] = path

    # ── Feature loading bar charts (PC1 & PC2) ────────────────────────

    for pc_num in [1, 2]:
        pc_col = f'PC{pc_num}'
        if pc_col in loadings_df.columns:
            path = output_dir / f'pca_loadings_{pc_col}_{mode}.png'
            plot_feature_loadings_bar(
                loadings_df, pc_col, path,
                title=f'{tp}{pc_col} Feature Loadings ({mode_label})',
            )
            output_files[f'pca_loadings_{pc_col}_{mode}'] = path

    # ── Feature loading 2-D scatter (PC1 vs PC2) ──────────────────────

    if 'PC1' in loadings_df.columns and 'PC2' in loadings_df.columns:
        path = output_dir / f'pca_feature_loadings_{mode}.png'
        plot_feature_loadings_scatter(
            loadings_df, path,
            title=f'{tp}Feature Loadings ({mode_label})',
        )
        output_files[f'pca_feature_loadings_{mode}'] = path

    return output_files
