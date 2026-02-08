"""
Plotting utilities for DNA Damage Panel analysis outputs.

Generates summary plots for CSV outputs produced by the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dna_damage_parquet_pipeline import parse_dilution_string


_META_COLUMNS = {
    "plate",
    "well",
    "genotype",
    "drug",
    "dilut_string",
    "dilut_um",
    "is_control",
    "moa",
    "dataset",
    "response_column",
    "model",
    "PC",
    "qc_pass",
}


@dataclass
class PlotResult:
    csv_path: Path
    plot_paths: List[Path]


class PlotGenerator:
    """Create plots for pipeline CSV outputs."""

    def __init__(self, output_dir: Path, logger: Optional[object] = None):
        self.output_dir = Path(output_dir)
        self.logger = logger

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger.info(message)

    @staticmethod
    def _load_csv(path: Path) -> pd.DataFrame:
        try:
            df = pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
        if df.empty:
            return df
        return df

    @staticmethod
    def _numeric_columns(df: pd.DataFrame, *, exclude: Iterable[str] = ()) -> List[str]:
        exclude_set = set(exclude)
        cols = [
            c
            for c in df.columns
            if c not in exclude_set and pd.api.types.is_numeric_dtype(df[c])
        ]
        return cols

    @staticmethod
    def _safe_filename(text: str) -> str:
        return text.replace("/", "_").replace("\\", "_")

    def _save_plot(self, fig: plt.Figure, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return path

    def _plot_metric_by_dose(
        self,
        df: pd.DataFrame,
        metric: str,
        dose_col: str,
        group_cols: List[str],
        output_path: Path,
    ) -> Optional[Path]:
        if metric not in df.columns or dose_col not in df.columns:
            return None

        plot_df = df[[dose_col, metric] + group_cols].dropna()
        if plot_df.empty:
            return None

        grouped = plot_df.groupby(group_cols + [dose_col])[metric].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        if group_cols:
            for name, group in grouped.groupby(group_cols):
                if not isinstance(name, tuple):
                    name = (name,)
                label = ", ".join(str(x) for x in name)
                sorted_group = group.sort_values(dose_col)
                ax.plot(sorted_group[dose_col], sorted_group[metric], marker="o", label=label)
            ax.legend(fontsize=7, loc="best")
        else:
            sorted_group = grouped.sort_values(dose_col)
            ax.plot(sorted_group[dose_col], sorted_group[metric], marker="o")
        ax.set_xlabel(dose_col)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs {dose_col}")
        return self._save_plot(fig, output_path)

    def _plot_hist_grid(self, df: pd.DataFrame, cols: List[str], output_path: Path) -> Optional[Path]:
        if not cols:
            return None
        n = min(len(cols), 6)
        cols = cols[:n]
        ncols = 2
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(7, 3.5 * nrows))
        axes = np.array(axes).reshape(-1)
        for ax, col in zip(axes, cols):
            ax.hist(df[col].dropna(), bins=30, color="#4C72B0", alpha=0.8)
            ax.set_title(col)
        for ax in axes[len(cols):]:
            ax.axis("off")
        return self._save_plot(fig, output_path)

    def _plot_scatter(self, df: pd.DataFrame, x_col: str, y_col: str, hue_col: Optional[str], output_path: Path) -> Optional[Path]:
        if x_col not in df.columns or y_col not in df.columns:
            return None
        plot_df = df[[x_col, y_col] + ([hue_col] if hue_col else [])].dropna()
        if plot_df.empty:
            return None
        fig, ax = plt.subplots(figsize=(6, 4.5))
        if hue_col and hue_col in plot_df.columns:
            for name, group in plot_df.groupby(hue_col):
                ax.scatter(group[x_col], group[y_col], alpha=0.7, label=str(name), s=20)
            ax.legend(fontsize=7, loc="best")
        else:
            ax.scatter(plot_df[x_col], plot_df[y_col], alpha=0.7, s=20)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{y_col} vs {x_col}")
        return self._save_plot(fig, output_path)

    def _plot_bar(self, labels: List[str], values: List[float], output_path: Path, title: str) -> Optional[Path]:
        if not labels:
            return None
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(labels, values, color="#55A868")
        ax.set_title(title)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        return self._save_plot(fig, output_path)

    def _plot_crowding(self, df: pd.DataFrame, plot_dir: Path, stem: str) -> List[Path]:
        if df.empty:
            return []
        if "dilut_um" not in df.columns and "dilut_string" in df.columns:
            df = df.copy()
            df["dilut_um"] = df["dilut_string"].apply(parse_dilution_string)
        dose_col = "dilut_um" if "dilut_um" in df.columns else None
        if dose_col is None:
            return []
        metric_cols = [
            c for c in df.columns
            if any(key in c for key in ("crowding", "nn_dist", "nbr_count", "mean_k"))
            and c.endswith(("_mean", "_median"))
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        metric_cols = metric_cols[:3]
        group_cols = [c for c in ["genotype", "drug", "dataset"] if c in df.columns]
        plot_paths = []
        for metric in metric_cols:
            out_path = plot_dir / f"{stem}_{self._safe_filename(metric)}.png"
            result = self._plot_metric_by_dose(df, metric, dose_col, group_cols, out_path)
            if result:
                plot_paths.append(result)
        return plot_paths

    def _plot_qc(self, df: pd.DataFrame, plot_dir: Path, stem: str) -> List[Path]:
        plot_paths = []
        if "qc_pass" in df.columns:
            counts = df["qc_pass"].value_counts()
            out_path = plot_dir / f"{stem}_qc_pass_counts.png"
            plot = self._plot_bar(
                [str(x) for x in counts.index],
                counts.values.tolist(),
                out_path,
                "QC Pass/Fail Counts",
            )
            if plot:
                plot_paths.append(plot)
        numeric_cols = self._numeric_columns(df, exclude=_META_COLUMNS)
        hist_path = plot_dir / f"{stem}_distributions.png"
        hist_plot = self._plot_hist_grid(df, numeric_cols, hist_path)
        if hist_plot:
            plot_paths.append(hist_plot)
        return plot_paths

    def _plot_pca_profiles(self, df: pd.DataFrame, plot_dir: Path, stem: str) -> List[Path]:
        if "PC1" not in df.columns or "PC2" not in df.columns:
            return []
        hue = "genotype" if "genotype" in df.columns else "drug" if "drug" in df.columns else None
        out_path = plot_dir / f"{stem}_pc1_pc2.png"
        plot = self._plot_scatter(df, "PC1", "PC2", hue, out_path)
        return [plot] if plot else []

    def _plot_pca_loadings(self, df: pd.DataFrame, plot_dir: Path, stem: str) -> List[Path]:
        if df.empty:
            return []
        if "PC1" not in df.columns:
            return []
        feature_col = df.columns[0]
        if feature_col.startswith("Unnamed"):
            df = df.rename(columns={feature_col: "feature"})
            feature_col = "feature"
        load_df = df[[feature_col, "PC1"]].copy()
        load_df = load_df.dropna()
        if load_df.empty:
            return []
        load_df["abs_pc1"] = load_df["PC1"].abs()
        top = load_df.nlargest(15, "abs_pc1")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(top[feature_col], top["PC1"], color="#C44E52")
        ax.set_title("Top PC1 Loadings")
        ax.invert_yaxis()
        out_path = plot_dir / f"{stem}_pc1_loadings.png"
        return [self._save_plot(fig, out_path)]

    def _plot_pca_variance(self, df: pd.DataFrame, plot_dir: Path, stem: str) -> List[Path]:
        if "PC" not in df.columns or "variance_ratio" not in df.columns:
            return []
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(df["PC"], df["variance_ratio"], color="#4C72B0", alpha=0.7, label="Variance ratio")
        if "cumulative_variance_ratio" in df.columns:
            ax.plot(df["PC"], df["cumulative_variance_ratio"], color="#DD8452", marker="o", label="Cumulative")
        ax.set_ylabel("Explained variance")
        ax.set_title("PCA Variance Explained")
        ax.legend()
        out_path = plot_dir / f"{stem}_variance.png"
        return [self._save_plot(fig, out_path)]

    def _plot_dose_response_fits(self, df: pd.DataFrame, plot_dir: Path, stem: str) -> List[Path]:
        if df.empty or "ec50" not in df.columns:
            return []
        plot_paths = []
        response_groups = df.groupby("response_column") if "response_column" in df.columns else [("response", df)]
        for response, group in response_groups:
            group = group.dropna(subset=["ec50"])
            if group.empty:
                continue
            labels = []
            for _, row in group.iterrows():
                parts = []
                for col in ["genotype", "drug"]:
                    if col in group.columns and pd.notna(row.get(col)):
                        parts.append(str(row[col]))
                labels.append(" ".join(parts) if parts else str(response))
            y = group["ec50"].values
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.errorbar(
                np.arange(len(y)),
                y,
                yerr=self._ec50_errors(group),
                fmt="o",
                capsize=4,
            )
            ax.set_xticks(np.arange(len(y)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_ylabel("EC50")
            ax.set_title(f"Dose-response EC50 ({response})")
            out_path = plot_dir / f"{stem}_{self._safe_filename(str(response))}_ec50.png"
            plot_paths.append(self._save_plot(fig, out_path))
        return plot_paths

    @staticmethod
    def _ec50_errors(df: pd.DataFrame) -> Optional[np.ndarray]:
        if "ec50_ci_lower" in df.columns and "ec50_ci_upper" in df.columns:
            lower = df["ec50"] - df["ec50_ci_lower"]
            upper = df["ec50_ci_upper"] - df["ec50"]
            return np.vstack([lower.values, upper.values])
        return None

    def _plot_dose_summary(self, df: pd.DataFrame, plot_dir: Path, stem: str) -> List[Path]:
        if "dilut_um" not in df.columns:
            return []
        mean_cols = [c for c in df.columns if c.endswith("_mean") and pd.api.types.is_numeric_dtype(df[c])]
        mean_cols = mean_cols[:3]
        group_cols = [c for c in ["genotype", "drug"] if c in df.columns]
        plot_paths = []
        for col in mean_cols:
            out_path = plot_dir / f"{stem}_{self._safe_filename(col)}.png"
            plot = self._plot_metric_by_dose(df, col, "dilut_um", group_cols, out_path)
            if plot:
                plot_paths.append(plot)
        return plot_paths

    def _plot_fold_change(self, df: pd.DataFrame, plot_dir: Path, stem: str) -> List[Path]:
        if "dilut_um" not in df.columns:
            return []
        fc_cols = [c for c in df.columns if c.endswith("_log2_fc")]
        if not fc_cols:
            fc_cols = [c for c in df.columns if c.endswith("_fold_change")]
        fc_cols = fc_cols[:3]
        group_cols = [c for c in ["genotype", "drug"] if c in df.columns]
        plot_paths = []
        for col in fc_cols:
            out_path = plot_dir / f"{stem}_{self._safe_filename(col)}.png"
            plot = self._plot_metric_by_dose(df, col, "dilut_um", group_cols, out_path)
            if plot:
                plot_paths.append(plot)
        return plot_paths

    def _plot_statistics(self, df: pd.DataFrame, plot_dir: Path, stem: str) -> List[Path]:
        plot_paths = []
        if "delta_log10_ec50" in df.columns:
            plot_df = df.dropna(subset=["delta_log10_ec50"])
            if not plot_df.empty:
                labels = []
                for _, row in plot_df.iterrows():
                    parts = [str(row.get(col)) for col in ["drug", "response_column"] if col in plot_df.columns]
                    parts.extend(
                        [
                            f"{row.get('genotype_1')}-{row.get('genotype_2')}"
                            if "genotype_1" in plot_df.columns
                            else ""
                        ]
                    )
                    labels.append(" ".join([p for p in parts if p]))
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.errorbar(
                    np.arange(len(plot_df)),
                    plot_df["delta_log10_ec50"],
                    yerr=self._delta_ec50_errors(plot_df),
                    fmt="o",
                    capsize=4,
                )
                ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
                ax.set_xticks(np.arange(len(plot_df)))
                ax.set_xticklabels(labels, rotation=45, ha="right")
                ax.set_ylabel("Δ log10(EC50)")
                ax.set_title("EC50 bootstrap comparisons")
                out_path = plot_dir / f"{stem}_delta_log10_ec50.png"
                plot_paths.append(self._save_plot(fig, out_path))
        if "effect_size_r" in df.columns and "dilut_um" in df.columns:
            plot = self._plot_metric_by_dose(
                df,
                "effect_size_r",
                "dilut_um",
                [c for c in ["drug"] if c in df.columns],
                plot_dir / f"{stem}_effect_size_r.png",
            )
            if plot:
                plot_paths.append(plot)
        if "log10_ec50_ratio" in df.columns:
            plot_df = df.dropna(subset=["log10_ec50_ratio"])
            if not plot_df.empty:
                labels = []
                for _, row in plot_df.iterrows():
                    parts = [str(row.get(col)) for col in ["drug"] if col in plot_df.columns]
                    if "genotype_1" in plot_df.columns and "genotype_2" in plot_df.columns:
                        parts.append(f"{row['genotype_1']}-{row['genotype_2']}")
                    labels.append(" ".join(parts))
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.bar(labels, plot_df["log10_ec50_ratio"], color="#4C72B0")
                ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
                ax.set_ylabel("log10(EC50 ratio)")
                ax.set_title("EC50 ratio comparisons")
                ax.set_xticklabels(labels, rotation=45, ha="right")
                out_path = plot_dir / f"{stem}_log10_ec50_ratio.png"
                plot_paths.append(self._save_plot(fig, out_path))
        return plot_paths

    @staticmethod
    def _delta_ec50_errors(df: pd.DataFrame) -> Optional[np.ndarray]:
        if "ci_lower" in df.columns and "ci_upper" in df.columns:
            lower = df["delta_log10_ec50"] - df["ci_lower"]
            upper = df["ci_upper"] - df["delta_log10_ec50"]
            return np.vstack([lower.values, upper.values])
        return None

    def _plot_generic(self, df: pd.DataFrame, plot_dir: Path, stem: str) -> List[Path]:
        numeric_cols = self._numeric_columns(df, exclude=_META_COLUMNS)
        if len(numeric_cols) >= 2:
            out_path = plot_dir / f"{stem}_scatter.png"
            plot = self._plot_scatter(df, numeric_cols[0], numeric_cols[1], None, out_path)
            return [plot] if plot else []
        if len(numeric_cols) == 1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(df[numeric_cols[0]].dropna(), bins=30, color="#4C72B0", alpha=0.8)
            ax.set_title(numeric_cols[0])
            out_path = plot_dir / f"{stem}_hist.png"
            return [self._save_plot(fig, out_path)]
        return []

    def generate_for_csv(self, csv_path: Path) -> PlotResult:
        df = self._load_csv(csv_path)
        plot_dir = self.output_dir / "plots" / csv_path.parent.name
        stem = csv_path.stem
        plot_paths: List[Path] = []

        if df.empty:
            return PlotResult(csv_path=csv_path, plot_paths=[])

        name = csv_path.name
        if "crowding" in name:
            plot_paths.extend(self._plot_crowding(df, plot_dir, stem))
        elif "well_qc_metrics" in name or "qc" in csv_path.parent.parts:
            plot_paths.extend(self._plot_qc(df, plot_dir, stem))
        elif "profiles_pca" in name:
            plot_paths.extend(self._plot_pca_profiles(df, plot_dir, stem))
        elif "pca_loadings" in name:
            plot_paths.extend(self._plot_pca_loadings(df, plot_dir, stem))
        elif "pca_variance" in name:
            plot_paths.extend(self._plot_pca_variance(df, plot_dir, stem))
        elif "dose_response_fits" in name:
            plot_paths.extend(self._plot_dose_response_fits(df, plot_dir, stem))
        elif "dose_response_summary" in name:
            plot_paths.extend(self._plot_dose_summary(df, plot_dir, stem))
        elif "fold_change_vs_control" in name:
            plot_paths.extend(self._plot_fold_change(df, plot_dir, stem))
        elif "ec50" in name or "comparison" in name:
            plot_paths.extend(self._plot_statistics(df, plot_dir, stem))
        elif "well_profiles" in name or "well_effects" in name:
            plot_paths.extend(self._plot_dose_summary(df, plot_dir, stem))
        else:
            plot_paths.extend(self._plot_generic(df, plot_dir, stem))

        if not plot_paths:
            plot_paths.extend(self._plot_generic(df, plot_dir, stem))

        return PlotResult(csv_path=csv_path, plot_paths=plot_paths)

    def generate_plots(self, csv_paths: Iterable[Path]) -> List[PlotResult]:
        results = []
        for csv_path in csv_paths:
            results.append(self.generate_for_csv(csv_path))
        return results
