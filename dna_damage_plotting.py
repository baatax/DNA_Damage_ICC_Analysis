"""Plotting utilities for DNA Damage Panel analysis outputs."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dna_damage_parquet_pipeline import parse_dilution_string


QC_FEATURES_DEFAULT = [
    "cell_count",
    "segmentation_coverage",
    "fraction_border_cells",
    "missing_feature_fraction",
]

CROWDING_FEATURES_DEFAULT = [
    "crowding_local_mean_mean",
    "crowding_local_max_mean",
    "nn_dist_px_mean",
]


@dataclass
class PlotResult:
    csv_path: Path
    plot_paths: list[Path]
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class PlotGeneratorConfig:
    qc_features: list[str] = field(default_factory=lambda: QC_FEATURES_DEFAULT.copy())
    crowding_features: list[str] = field(default_factory=lambda: CROWDING_FEATURES_DEFAULT.copy())


class PlotGenerator:
    """Create plots for pipeline CSV outputs."""

    def __init__(
        self,
        output_dir: Path,
        logger: logging.Logger | None = None,
        config: PlotGeneratorConfig | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.logger = logger
        self.config = config or PlotGeneratorConfig()

        self.handlers: list[tuple[Callable[[Path], bool], Callable[[Path, pd.DataFrame], list[Path]]]] = [
            (lambda p: p.name == "well_qc_metrics.csv", self._handle_well_qc_metrics),
            (lambda p: p.name == "excluded_wells.csv", self._handle_excluded_wells),
            (lambda p: "crowding" in p.name, self._handle_crowding),
            (lambda p: p.name.startswith("profiles_pca_"), self._handle_pca_profiles),
            (lambda p: p.name.startswith("pca_variance_"), self._handle_pca_variance),
            (lambda p: p.name.startswith("pca_loadings_"), self._handle_pca_loadings),
            (lambda p: p.name == "dose_response_summary.csv", self._handle_dose_response_summary),
            (lambda p: p.name == "dose_response_fits.csv", self._handle_dose_response_fits),
            (lambda p: p.name == "fold_change_vs_control.csv", self._handle_fold_change),
            (lambda p: p.name.startswith("ec50_bootstrap_"), self._handle_ec50_bootstrap),
            (lambda p: p.name.startswith("genotype_comparison_"), self._handle_genotype_comparison),
            (lambda p: p.name.startswith("well_profiles_"), self._handle_well_profiles),
            (lambda p: p.name.startswith("well_effects_"), self._handle_well_effects),
        ]

    def _log(self, msg: str) -> None:
        if self.logger:
            self.logger.info(msg)

    def _warn(self, msg: str) -> None:
        if self.logger:
            self.logger.warning(msg)

    @staticmethod
    def _load_csv(path: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()

    @staticmethod
    def _safe_name(text: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")

    def _save(self, fig: plt.Figure, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return path

    def _require_columns(self, df: pd.DataFrame, cols: list[str], context: str) -> bool:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            self._warn(f"Skipping {context}: missing columns {missing}")
            return False
        return True

    @staticmethod
    def _well_to_row_col(well: str) -> tuple[int, int] | None:
        match = re.match(r"^([A-Za-z]+)\s*0*([0-9]+)$", str(well).strip())
        if not match:
            return None
        letters, col_str = match.groups()
        row = 0
        for ch in letters.upper():
            row = row * 26 + (ord(ch) - ord("A") + 1)
        return row - 1, int(col_str) - 1

    @staticmethod
    def _as_numeric_dose(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "dilut_um" not in out.columns and "dilut_string" in out.columns:
            out["dilut_um"] = out["dilut_string"].apply(parse_dilution_string)
        return out

    def _plot_plate_heatmap(self, plate_df: pd.DataFrame, value_col: str, out_path: Path) -> Optional[Path]:
        if not self._require_columns(plate_df, ["well", value_col], f"plate heatmap ({value_col})"):
            return None
        parsed = plate_df["well"].map(self._well_to_row_col)
        valid = parsed.notna()
        if not valid.any():
            self._warn(f"Skipping plate heatmap ({value_col}): could not parse well IDs")
            return None

        tmp = plate_df.loc[valid, ["well", value_col]].copy()
        coords = parsed[valid]
        tmp["row"] = coords.map(lambda x: x[0])
        tmp["col"] = coords.map(lambda x: x[1])
        nrows = int(tmp["row"].max()) + 1
        ncols = int(tmp["col"].max()) + 1
        mat = np.full((nrows, ncols), np.nan)
        for _, row in tmp.iterrows():
            mat[int(row["row"]), int(row["col"])] = row[value_col]

        fig, ax = plt.subplots(figsize=(max(6, ncols * 0.5), max(3, nrows * 0.5)))
        im = ax.imshow(mat, aspect="auto", cmap="viridis")
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(value_col)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        return self._save(fig, out_path)

    def _handle_well_qc_metrics(self, csv_path: Path, df: pd.DataFrame) -> list[Path]:
        paths: list[Path] = []
        qc_dir = self.output_dir / "plots" / "qc"

        # distributions with threshold overlays
        features = [c for c in self.config.qc_features if c in df.columns]
        if features:
            ncols = 2
            nrows = int(np.ceil(len(features) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(9, 3.8 * nrows))
            axes = np.array(axes).reshape(-1)
            for idx, feat in enumerate(features):
                ax = axes[idx]
                ax.hist(df[feat].dropna(), bins=30, color="#4C72B0", alpha=0.85)
                ax.set_title(feat)
                if feat == "cell_count":
                    for tag in ("n_cells_min", "n_cells_max"):
                        if tag in df.columns:
                            val = df[tag].dropna().iloc[0] if not df[tag].dropna().empty else None
                            if val is not None:
                                ax.axvline(val, color="red", linestyle="--", linewidth=1)
                elif feat == "segmentation_coverage" and "coverage_min" in df.columns:
                    vals = df["coverage_min"].dropna()
                    if not vals.empty:
                        ax.axvline(vals.iloc[0], color="red", linestyle="--", linewidth=1)
                elif feat == "fraction_border_cells" and "border_max" in df.columns:
                    vals = df["border_max"].dropna()
                    if not vals.empty:
                        ax.axvline(vals.iloc[0], color="red", linestyle="--", linewidth=1)
                elif feat == "missing_feature_fraction" and "missing_max" in df.columns:
                    vals = df["missing_max"].dropna()
                    if not vals.empty:
                        ax.axvline(vals.iloc[0], color="red", linestyle="--", linewidth=1)
            for ax in axes[len(features):]:
                ax.axis("off")
            paths.append(self._save(fig, qc_dir / "well_qc_metrics_distributions.png"))

        # qc pass/fail counts by genotype / drug
        if "qc_pass" in df.columns:
            group_cols = [c for c in ["genotype", "drug"] if c in df.columns]
            if group_cols:
                agg = (
                    df.groupby(group_cols + ["qc_pass"]).size().reset_index(name="n")
                )
                fig, axes = plt.subplots(1, len(group_cols), figsize=(6 * len(group_cols), 4), squeeze=False)
                for i, gcol in enumerate(group_cols):
                    ax = axes[0, i]
                    pivot = agg.pivot_table(index=gcol, columns="qc_pass", values="n", fill_value=0)
                    pivot.plot(kind="bar", stacked=True, ax=ax, color=["#C44E52", "#55A868"])
                    ax.set_title(f"QC pass/fail by {gcol}")
                    ax.set_ylabel("Wells")
                    ax.tick_params(axis="x", labelrotation=45)
                paths.append(self._save(fig, qc_dir / "well_qc_metrics_qc_pass_counts.png"))

        # plate heatmaps
        if "plate" in df.columns and "well" in df.columns:
            for plate, plate_df in df.groupby("plate"):
                plate_key = self._safe_name(plate)
                if "qc_pass" in plate_df.columns:
                    tmp = plate_df.copy()
                    tmp["qc_pass_numeric"] = tmp["qc_pass"].astype(int)
                    p = self._plot_plate_heatmap(tmp, "qc_pass_numeric", qc_dir / f"qc_pass_plate_{plate_key}.png")
                    if p:
                        paths.append(p)
                if "cell_count" in plate_df.columns:
                    p = self._plot_plate_heatmap(plate_df, "cell_count", qc_dir / f"cell_count_plate_{plate_key}.png")
                    if p:
                        paths.append(p)
                if "segmentation_coverage" in plate_df.columns:
                    p = self._plot_plate_heatmap(plate_df, "segmentation_coverage", qc_dir / f"coverage_plate_{plate_key}.png")
                    if p:
                        paths.append(p)

        # segmentation proxies
        if self._require_columns(df, ["cell_count", "segmentation_coverage"], "segmentation scatter"):
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            if "qc_pass" in df.columns:
                for val, sub in df.groupby("qc_pass"):
                    ax.scatter(sub["cell_count"], sub["segmentation_coverage"], s=18, alpha=0.7, label=f"qc_pass={val}")
                ax.legend(fontsize=8)
            else:
                ax.scatter(df["cell_count"], df["segmentation_coverage"], s=18, alpha=0.7)
            ax.set_xlabel("cell_count")
            ax.set_ylabel("segmentation_coverage")
            ax.set_title("Cell count vs segmentation coverage")
            paths.append(self._save(fig, qc_dir / "segmentation_scatter_count_vs_coverage.png"))

        area_col = "cell_area_mean" if "cell_area_mean" in df.columns else "area_mean" if "area_mean" in df.columns else None
        if area_col and "cell_count" in df.columns:
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            ax.scatter(df[area_col], df["cell_count"], s=18, alpha=0.7)
            ax.set_xlabel(area_col)
            ax.set_ylabel("cell_count")
            ax.set_title(f"Cell area proxy vs cell count ({area_col})")
            paths.append(self._save(fig, qc_dir / "segmentation_scatter_area_vs_count.png"))

        return paths

    def _handle_excluded_wells(self, csv_path: Path, df: pd.DataFrame) -> list[Path]:
        if "qc_fail_reasons" not in df.columns:
            return []
        reasons = (
            df["qc_fail_reasons"].fillna("").str.split(";").explode().str.strip()
        )
        reasons = reasons[reasons != ""]
        if reasons.empty:
            return []
        counts = reasons.value_counts()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(counts.index.astype(str), counts.values, color="#C44E52")
        ax.set_title("Exclusion reasons")
        ax.set_ylabel("Wells")
        ax.tick_params(axis="x", labelrotation=45)
        return [self._save(fig, self.output_dir / "plots" / "qc" / "exclusion_reasons_bar.png")]

    def _handle_crowding(self, csv_path: Path, df: pd.DataFrame) -> list[Path]:
        out: list[Path] = []
        df = self._as_numeric_dose(df)
        if "dilut_um" not in df.columns:
            return out
        group_cols = [c for c in ["drug", "genotype"] if c in df.columns]
        for metric in [c for c in self.config.crowding_features if c in df.columns]:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            grouped = df[group_cols + ["dilut_um", metric]].dropna().groupby(group_cols + ["dilut_um"], as_index=False)[metric].mean()
            if grouped.empty:
                continue
            if group_cols:
                for name, sub in grouped.groupby(group_cols):
                    label = name if isinstance(name, str) else " | ".join(map(str, name))
                    sub = sub.sort_values("dilut_um")
                    ax.plot(sub["dilut_um"], sub[metric], marker="o", label=label)
            else:
                grouped = grouped.sort_values("dilut_um")
                ax.plot(grouped["dilut_um"], grouped[metric], marker="o")
            pos = grouped["dilut_um"] > 0
            if pos.any():
                ax.set_xscale("log")
            ax.set_xlabel("dilut_um")
            ax.set_ylabel(metric)
            ax.set_title(f"Crowding vs dose: {metric}")
            if group_cols:
                ax.legend(fontsize=7)
            out.append(self._save(fig, self.output_dir / "plots" / "crowding" / f"crowding_by_drug_dose_{self._safe_name(metric)}.png"))
        return out

    def _mode_from_name(self, name: str) -> str:
        for mode in ["across_all", "per_genotype"]:
            if mode in name:
                return mode
        return "across_all"

    def _handle_well_profiles(self, csv_path: Path, df: pd.DataFrame) -> list[Path]:
        mode = self._mode_from_name(csv_path.stem)
        plot_dir = self.output_dir / "plots" / mode
        # dose series for gamma_h2ax if present
        out: list[Path] = []
        df = self._as_numeric_dose(df)
        feature = "gamma_h2ax_nuclear_mean"
        if {"dilut_um", feature}.issubset(df.columns):
            fig, ax = plt.subplots(figsize=(7, 4.5))
            gcols = [c for c in ["genotype", "drug"] if c in df.columns]
            grp = df[gcols + ["dilut_um", feature]].dropna().groupby(gcols + ["dilut_um"], as_index=False)[feature].median()
            for name, sub in grp.groupby(gcols):
                label = name if isinstance(name, str) else " | ".join(map(str, name))
                sub = sub.sort_values("dilut_um")
                ax.plot(sub["dilut_um"], sub[feature], marker="o", label=label)
            if (grp["dilut_um"] > 0).any():
                ax.set_xscale("log")
            ax.set_xlabel("dilut_um")
            ax.set_ylabel(feature)
            ax.set_title(f"Well profiles {feature} ({mode})")
            if gcols:
                ax.legend(fontsize=7)
            out.append(self._save(fig, plot_dir / f"well_profiles_ALL_{mode}_{feature}.png"))

        # correlation heatmap
        numeric = df.select_dtypes(include=[np.number]).copy()
        for c in ["dilut_um", "is_control"]:
            if c in numeric.columns:
                numeric = numeric.drop(columns=[c])
        if numeric.shape[1] >= 2:
            corr = numeric.corr(method="spearman")
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
            fig.colorbar(im, ax=ax, shrink=0.8)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
            ax.set_yticks(range(len(corr.columns)))
            ax.set_yticklabels(corr.columns, fontsize=7)
            ax.set_title(f"Well feature correlation ({mode})")
            out.append(self._save(fig, self.output_dir / "plots" / "statistics" / f"well_feature_correlation_heatmap_{mode}.png"))
        return out

    def _handle_well_effects(self, csv_path: Path, df: pd.DataFrame) -> list[Path]:
        mode = self._mode_from_name(csv_path.stem)
        plot_dir = self.output_dir / "plots" / mode
        out: list[Path] = []
        df = self._as_numeric_dose(df)
        metric = "gamma_h2ax_nuclear_mean_delta"
        if {"dilut_um", metric}.issubset(df.columns):
            fig, ax = plt.subplots(figsize=(7, 4.5))
            gcols = [c for c in ["genotype", "drug"] if c in df.columns]
            for name, sub in df[gcols + ["dilut_um", metric]].dropna().groupby(gcols):
                label = name if isinstance(name, str) else " | ".join(map(str, name))
                q = sub.groupby("dilut_um")[metric].median().reset_index().sort_values("dilut_um")
                ax.plot(q["dilut_um"], q[metric], marker="o", label=label)
            if (df["dilut_um"] > 0).any():
                ax.set_xscale("log")
            ax.set_xlabel("dilut_um")
            ax.set_ylabel(metric)
            ax.set_title(f"Well effects {metric} ({mode})")
            if gcols:
                ax.legend(fontsize=7)
            out.append(self._save(fig, plot_dir / f"well_effects_ALL_{mode}_gamma_h2ax_nuclear_mean.png"))
        return out

    def _handle_pca_profiles(self, csv_path: Path, df: pd.DataFrame) -> list[Path]:
        if not self._require_columns(df, ["PC1", "PC2"], "PCA scatter"):
            return []
        mode = self._mode_from_name(csv_path.stem)
        plot_dir = self.output_dir / "plots" / mode
        out: list[Path] = []

        # color by genotype
        fig, ax = plt.subplots(figsize=(6.5, 5))
        color_col = "genotype" if "genotype" in df.columns else None
        if color_col:
            for key, sub in df.groupby(color_col):
                ax.scatter(sub["PC1"], sub["PC2"], s=18, alpha=0.7, label=str(key))
            ax.legend(fontsize=8)
        else:
            ax.scatter(df["PC1"], df["PC2"], s=18, alpha=0.7)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"PCA profiles ({mode})")
        out.append(self._save(fig, plot_dir / f"profiles_pca_{mode}_pc1_pc2.png"))

        # color by plate for QC confounding check
        if "plate" in df.columns:
            fig, ax = plt.subplots(figsize=(6.5, 5))
            for key, sub in df.groupby("plate"):
                ax.scatter(sub["PC1"], sub["PC2"], s=16, alpha=0.65, label=str(key))
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title(f"PCA by plate ({mode})")
            if df["plate"].nunique() <= 15:
                ax.legend(fontsize=6)
            out.append(self._save(fig, self.output_dir / "plots" / "qc" / f"pca_pc1_pc2_colored_by_plate_{mode}.png"))
        return out

    def _handle_pca_variance(self, csv_path: Path, df: pd.DataFrame) -> list[Path]:
        if not self._require_columns(df, ["PC", "variance_ratio"], "PCA variance"):
            return []
        mode = self._mode_from_name(csv_path.stem)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(df["PC"], df["variance_ratio"], label="variance_ratio", color="#4C72B0")
        if "cumulative_variance_ratio" in df.columns:
            ax.plot(df["PC"], df["cumulative_variance_ratio"], marker="o", color="#DD8452", label="cumulative")
            ax.legend()
        ax.set_ylabel("Explained variance")
        ax.set_title(f"PCA variance ({mode})")
        return [self._save(fig, self.output_dir / "plots" / mode / f"pca_variance_{mode}_variance.png")]

    def _handle_pca_loadings(self, csv_path: Path, df: pd.DataFrame) -> list[Path]:
        mode = self._mode_from_name(csv_path.stem)
        if "PC1" not in df.columns:
            return []
        feature_col = "feature" if "feature" in df.columns else df.columns[0]
        if feature_col.startswith("Unnamed"):
            df = df.rename(columns={feature_col: "feature"})
            feature_col = "feature"
        use = df[[feature_col, "PC1"]].dropna().copy()
        if use.empty:
            return []
        use["abs_pc1"] = use["PC1"].abs()
        top = use.nlargest(20, "abs_pc1").sort_values("PC1")
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.barh(top[feature_col].astype(str), top["PC1"], color="#55A868")
        ax.set_title(f"Top PC1 loadings ({mode})")
        return [self._save(fig, self.output_dir / "plots" / mode / f"pca_loadings_{mode}_pc1_loadings.png")]

    def _response_columns(self, df: pd.DataFrame) -> list[str]:
        resp = []
        for c in df.columns:
            if c.endswith("_mean") and c not in {"dilut_um_mean"}:
                resp.append(c)
        return resp

    def _handle_dose_response_summary(self, csv_path: Path, df: pd.DataFrame) -> list[Path]:
        df = self._as_numeric_dose(df)
        if "dilut_um" not in df.columns:
            return []
        out: list[Path] = []
        for resp in self._response_columns(df)[:4]:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            gcols = [c for c in ["genotype", "drug"] if c in df.columns]
            use = df[gcols + ["dilut_um", resp]].dropna()
            for name, sub in use.groupby(gcols):
                label = name if isinstance(name, str) else " | ".join(map(str, name))
                sub = sub.sort_values("dilut_um")
                ax.plot(sub["dilut_um"], sub[resp], marker="o", label=label)
            if (use["dilut_um"] > 0).any():
                ax.set_xscale("log")
            ax.set_xlabel("dilut_um")
            ax.set_ylabel(resp)
            ax.set_title(f"Dose response summary: {resp}")
            if gcols:
                ax.legend(fontsize=7)
            out.append(self._save(fig, self.output_dir / "plots" / "dose_response" / f"dose_response_summary_{self._safe_name(resp)}.png"))
        return out

    def _handle_dose_response_fits(self, csv_path: Path, df: pd.DataFrame) -> list[Path]:
        if "ec50" not in df.columns:
            return []
        out: list[Path] = []
        group_cols = [c for c in ["response_column", "drug", "genotype"] if c in df.columns]
        if not group_cols:
            group_cols = [df.columns[0]]
        for response, sub in df.groupby("response_column") if "response_column" in df.columns else [("response", df)]:
            p = sub.dropna(subset=["ec50"]).copy()
            if p.empty:
                continue
            p = p.sort_values("ec50")
            labels = [" | ".join(str(r.get(c, "")) for c in ["drug", "genotype"] if c in p.columns).strip(" |") for _, r in p.iterrows()]
            x = np.arange(len(p))
            y = p["ec50"].values
            fig, ax = plt.subplots(figsize=(max(7, len(p) * 0.5), 4.5))
            if {"ec50_ci_lower", "ec50_ci_upper"}.issubset(p.columns):
                lower = y - p["ec50_ci_lower"].values
                upper = p["ec50_ci_upper"].values - y
                ax.errorbar(x, y, yerr=np.vstack([lower, upper]), fmt="o", capsize=4)
            else:
                ax.plot(x, y, "o")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_ylabel("EC50")
            ax.set_title(f"Dose-response fits ({response})")
            out.append(self._save(fig, self.output_dir / "plots" / "dose_response" / f"dose_response_fits_{self._safe_name(response)}_ec50.png"))
        return out

    def _handle_fold_change(self, csv_path: Path, df: pd.DataFrame) -> list[Path]:
        df = self._as_numeric_dose(df)
        if "dilut_um" not in df.columns:
            return []
        cols = [c for c in df.columns if c.endswith("_log2_fc")][:4]
        out: list[Path] = []
        for col in cols:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            gcols = [c for c in ["genotype", "drug"] if c in df.columns]
            use = df[gcols + ["dilut_um", col]].dropna()
            for name, sub in use.groupby(gcols):
                label = name if isinstance(name, str) else " | ".join(map(str, name))
                sub = sub.sort_values("dilut_um")
                ax.plot(sub["dilut_um"], sub[col], marker="o", label=label)
            if (use["dilut_um"] > 0).any():
                ax.set_xscale("log")
            ax.set_xlabel("dilut_um")
            ax.set_ylabel(col)
            ax.set_title(f"Fold-change vs control: {col}")
            if gcols:
                ax.legend(fontsize=7)
            out.append(self._save(fig, self.output_dir / "plots" / "dose_response" / f"fold_change_vs_control_{self._safe_name(col)}.png"))
        return out

    def _handle_ec50_bootstrap(self, csv_path: Path, df: pd.DataFrame) -> list[Path]:
        if "delta_log10_ec50" not in df.columns:
            return []
        response = csv_path.stem.replace("ec50_bootstrap_", "")
        vals = df["delta_log10_ec50"].dropna()
        if vals.empty:
            return []
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(vals, bins=30, color="#4C72B0", alpha=0.8)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        if {"ci_lower", "ci_upper"}.issubset(df.columns):
            ci_low = df["ci_lower"].dropna().iloc[0] if not df["ci_lower"].dropna().empty else None
            ci_up = df["ci_upper"].dropna().iloc[0] if not df["ci_upper"].dropna().empty else None
            if ci_low is not None and ci_up is not None:
                ax.axvline(ci_low, color="red", linestyle=":")
                ax.axvline(ci_up, color="red", linestyle=":")
        ax.set_title(f"Bootstrap Δlog10(EC50): {response}")
        ax.set_xlabel("delta_log10_ec50")
        return [self._save(fig, self.output_dir / "plots" / "statistics" / f"ec50_bootstrap_{self._safe_name(response)}_delta_log10_ec50.png")]

    def _handle_genotype_comparison(self, csv_path: Path, df: pd.DataFrame) -> list[Path]:
        out: list[Path] = []
        response = csv_path.stem.replace("genotype_comparison_", "")
        if {"log10_ec50_ratio", "drug"}.issubset(df.columns):
            fig, ax = plt.subplots(figsize=(8, 4.5))
            x = np.arange(len(df))
            ax.bar(x, df["log10_ec50_ratio"], color="#55A868")
            ax.axhline(0, color="black", linestyle="--", linewidth=1)
            labels = df["drug"].astype(str)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_ylabel("log10(EC50 ratio)")
            ax.set_title(f"EC50 forest proxy: {response}")
            out.append(self._save(fig, self.output_dir / "plots" / "statistics" / f"ec50_forest_{self._safe_name(response)}.png"))
        return out

    def _fallback_generic(self, csv_path: Path, df: pd.DataFrame) -> list[Path]:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric) < 2:
            return []
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.scatter(df[numeric[0]], df[numeric[1]], s=16, alpha=0.7)
        ax.set_xlabel(numeric[0])
        ax.set_ylabel(numeric[1])
        ax.set_title(csv_path.stem)
        subdir = csv_path.parent.name
        return [self._save(fig, self.output_dir / "plots" / subdir / f"{csv_path.stem}_scatter.png")]

    def generate_for_csv(self, csv_path: Path) -> PlotResult:
        df = self._load_csv(csv_path)
        if df.empty:
            self._warn(f"Skipping empty or unreadable CSV: {csv_path}")
            return PlotResult(csv_path=csv_path, plot_paths=[], tags={"status": "empty"})

        for matcher, handler in self.handlers:
            if matcher(csv_path):
                plots = handler(csv_path, df)
                if plots:
                    return PlotResult(csv_path=csv_path, plot_paths=plots, tags={"handler": handler.__name__})
                break

        generic = self._fallback_generic(csv_path, df)
        if generic:
            return PlotResult(csv_path=csv_path, plot_paths=generic, tags={"handler": "fallback"})
        return PlotResult(csv_path=csv_path, plot_paths=[], tags={"status": "skipped"})

    def generate_plots(self, csv_paths: Iterable[Path]) -> list[PlotResult]:
        results: list[PlotResult] = []
        seen: set[Path] = set()
        for path in csv_paths:
            p = Path(path)
            if p in seen:
                continue
            seen.add(p)
            results.append(self.generate_for_csv(p))
        return results
