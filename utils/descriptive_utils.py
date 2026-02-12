from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from pandas import DataFrame

from scipy import stats


def _iqr_bounds(x: pd.Series, k: float = 1.5) -> Tuple[float, float]:
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return float(lower), float(upper)


def summarize(df: pd.DataFrame, var: str,shapiro: bool = True, to_print=True) -> pd.Series:
    if var not in df.columns:
        raise KeyError(f"Variable '{var}' nicht in df.")

    x = df[var]
    n = len(x)
    desc = x.describe()
    iqr = desc["75%"] - desc["25%"]
    span = desc["max"] - desc["min"]
    cv = desc["std"] / desc["mean"]
    std = x.std(ddof=1)
    skew = float(stats.skew(x, bias=False)) if n > 2 else np.nan
    kurt = float(stats.kurtosis(x, fisher=True, bias=False)) if n > 3 else np.nan

    summary = {
        "n": n,
        "mean": desc["mean"],
        "median": desc["50%"],
        "std": std,
        "max": desc["max"],
        "min": desc["min"],
        "IQR": iqr,
        "span": span,
        "IQR/span": iqr / span,
        "CV": cv,
        "Skewness": skew,
        "Kurtosis": kurt,
    }

    if shapiro and 3 <= n <= 100:
        sh = stats.shapiro(x)
        summary["Shapiro-Statistik"] = float(sh.statistic)
        summary["mit p-value"] = float(sh.pvalue)

    out = pd.Series(summary, name=f"{var}").round(3)

    out.index.name = "Variable"

    path = Path(f"../../tables/{var}")
    path.mkdir(parents=True, exist_ok=True)

    if to_print:
        out.to_csv(path / f"{var}_summary.csv")
        return out
    else:
        return out


def grouped_summary(
    df: pd.DataFrame,
    var: str,
    group_col: str,
    shapiro: bool = True
) -> None:


    if var not in df.columns:
        raise KeyError(f"Variable '{var}' nicht in df.")

    series_by_group = (
        df.groupby(group_col, dropna=False)[var]
        .apply(lambda s: summarize(s.to_frame(name=var), var=var, shapiro=shapiro, to_print=False))
    )

    out = series_by_group.unstack(level=group_col)
    out.index.name = "Strukturvariable"

    out.to_csv(Path(f"../../tables/{var}") / f"{var}_by_{group_col}_summary.csv")


# Outlier
def outlier_table(
    df: pd.DataFrame,
    var: str,
    id_col: Optional[str] = "Name",
    iqr_k: float = 1.5,
) -> None:
    x = df[var]
    lower, upper = _iqr_bounds(x, k=iqr_k)

    mask_lower = df[var] < lower
    mask_upper = df[var] > upper

    cols = []
    if id_col and id_col in df.columns:
        cols.append(id_col)
    cols.append(var)

    out = df.loc[mask_lower | mask_upper, cols].copy()

    out["Outlier-Typ"] = np.where(
        mask_lower.loc[out.index],
        "unterer Outlier",
        "oberer Outlier",
    )

    out.sort_values(["Outlier-Typ", var]).reset_index(drop=True)


    path = Path(f"../../tables/{var}")
    path.mkdir(parents=True, exist_ok=True)
    out.to_csv(
        path / f"{var}_outlier.csv",
        index=False
    )


def clean_outlier(df, var, iqr_k: float = 1.5,) -> DataFrame:
    x = df[var]
    lower, upper = _iqr_bounds(x, k=iqr_k)

    df.loc[df[var] < lower, var] = lower
    df.loc[df[var] > upper, var] = upper

    return df

# Dist-Plot mit KDE
def plot_distribution(
    df: pd.DataFrame,
    var: str,
    bins: int = 20,
    bw_method: float = 0.5,
) -> None:

    x = df[var]
    xx = np.linspace(x.min(), x.max(), 200)

    plt.figure()
    plt.hist(x, bins=bins, density=True)
    plt.title(f"Verteilung: {var}")
    plt.xlabel(var)
    plt.ylabel("Häufigkeit")

    kde = stats.gaussian_kde(x.to_numpy(), bw_method=bw_method)
    plt.plot(xx, kde(xx), color="red")

    path = Path(f"../../figures/{var}")
    path.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        path / f"{var}_dist.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

# Boxplot nach Gruppen
def boxplot_by_group(
    df: pd.DataFrame,
    var: str,
    group_col: str,
) -> None:

    d = df[[group_col, var]].copy()

    groups = []
    labels = []
    for key, g in d.groupby(group_col, dropna=False):
        x = g[var].dropna()
        if len(x) == 0:
            continue
        groups.append(x.values)
        labels.append(str(key))

    plt.figure(figsize=(max(6, 0.6 * len(labels)), 6))
    plt.title(f"{var} nach {group_col}")
    sns.boxplot(
        data=df,
        x=group_col,
        y=var,
        showfliers=False
    )
    sns.stripplot(
        data=df,
        x=group_col,
        y=var,
        color="black",
        alpha=0.5,
        jitter=True
    )
    plt.ylabel(var)
    plt.xlabel(group_col)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    path = Path(f"../../figures/{var}")
    path.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        path / f"{var}_by_{group_col}_boxplot.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

# Scatterplot
def plot_scatter(
    df: pd.DataFrame,
    x: str,
    y: str="35a Hilfen pro 10000",
    hue: Optional[str] = None,
) -> None:
    if x == y:
        return None
    d = df.copy()

    plt.figure()
    if hue is None:
        plt.scatter(d[x], d[y])
    else:
        for key, g in d.groupby(hue, dropna=False):
            plt.scatter(g[x], g[y], label=str(key))
        plt.legend(title=hue)

    plt.title(f"Scatterplot von {x} und {y}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()

    path = Path(f"../../figures/{x}")
    path.mkdir(parents=True, exist_ok=True)

    if hue is not None:
        plt.savefig(
            path / f"{x}_vs_{y}__by_{hue}_scatter.png",
            dpi=300,
            bbox_inches="tight"
        )
    else:
        plt.savefig(
            path / f"{x}_vs_{y}_scatter.png",
            dpi=300,
            bbox_inches="tight"
        )
    plt.close()


# Choroplethenkarte
def show_map(df,
             var,
             group_col: str=None):
    nrw = gpd.read_file("../../data/NRW-map/dvg1krs_nw.shp")

    mapping = {
        "Mülheim a.d. Ruhr": "Mülheim an der Ruhr",
        'Städteregion Aachen': 'Aachen'
    }

    nrw.loc[:, "GN"] = (
        nrw["GN"].replace(mapping)
    )

    nrw = nrw.merge(df, left_on="GN", right_on="Name", how="left")

    fig, ax = plt.subplots(1, 1, figsize=(18, 21))
    nrw.plot(column=var, ax=ax, legend=True, cmap="OrRd", edgecolor="black")

    if group_col is not None:
        groups = nrw[group_col].unique().tolist()

        baseline = nrw[group_col].astype(str).value_counts().idxmax()
        other_groups = [g for g in groups if g != baseline]

        # Default-Hatches
        hatches = [".", "/", "\\", "o", "x", "+", "*", "-", "|", "O", "X"]

        if len(hatches) < len(other_groups):
            raise ValueError(
                f"Zu wenige hatches: {len(hatches)} für {len(other_groups)} Gruppen (n-1). "
                f"Bitte mehr Muster übergeben."
            )

        for grp, hatch in zip(other_groups, hatches):
            nrw[nrw[group_col].astype(str) == grp].plot(
                ax=ax,
                facecolor="none",
                edgecolor="black",
                hatch=hatch,
                linewidth=0,
            )

    for idx, row in nrw.iterrows():
        x, y = row['geometry'].centroid.x, row['geometry'].centroid.y
        ax.text(x, y, row['GN'], fontsize=8, ha='center', va='center')

    path = Path(f"../../figures/{var}")
    path.mkdir(parents=True, exist_ok=True)

    if group_col is None:
        plt.savefig(
            path / f"{var}_nrw_map.png",
            dpi=300,
            bbox_inches="tight"
        )
    else:
        plt.savefig(
            path / f"{var}_by_{group_col}_nrw_map.png",
            dpi=300,
            bbox_inches="tight"
        )
    plt.close()

# Korrelation
def corr_pair(
    df: pd.DataFrame,
    x: str,
    y: str="35a Hilfen pro 10000",
    to_print=True
) -> pd.DataFrame | None:
    if x == y:
        return None
    dx = df[x]
    dy = df[y]
    mask = dx.notna() & dy.notna()
    x2 = dx[mask]
    y2 = dy[mask]

    if len(x2) < 3:
        return pd.DataFrame([{
            "x": x,
            "pearson_r": np.nan, "pearson_p": np.nan,
            "spearman_r": np.nan, "spearman_p": np.nan,
        }])

    pr = stats.pearsonr(x2, y2)
    sr = stats.spearmanr(x2, y2)

    out= pd.DataFrame([{
        "pearson_r": float(pr.statistic), "pearson_p": float(pr.pvalue),
        "spearman_r": float(sr.statistic), "spearman_p": float(sr.pvalue),
    }])

    path = Path(f"../../tables/{x}")
    path.mkdir(parents=True, exist_ok=True)

    if to_print:
        out.to_csv(path / f"{x}_to_{y}_correlation.csv")
        return out
    else:
        return out



def corr_pair_by_type(
    df: pd.DataFrame,
    x: str,
    group: str,
    y: str="35a Hilfen pro 10000",
) -> DataFrame | None:
    if x == y:
        return None
    out = (df
        .groupby(group)
        .apply(lambda g: corr_pair(g, x, y, to_print=False), include_groups=False)
        .reset_index(level=0)
    )
    out.to_csv(Path(f"../../tables/{x}") / f"{x}_to_{y}_by_{group}_correlation.csv")
    return out


def basic_descriptive_analysis(df, var, y="35a Hilfen pro 10000", shapiro=True) -> None:

    summarize(df, var, shapiro=shapiro),
    outlier_table(df, var)
    plot_distribution(df, var)

    corr_pair(df, var, y)
    plot_scatter(df, var, y)

    show_map(df, var)


def descriptive_analysis_by_group(df, var,  group_col, y="35a Hilfen pro 10000", shapiro=True)-> None:

    grouped_summary(df, var, group_col, shapiro=shapiro)

    corr_pair_by_type(df, var, group_col, y)

    boxplot_by_group(df, var, group_col)

    plot_scatter(df, var, y, hue=group_col)




