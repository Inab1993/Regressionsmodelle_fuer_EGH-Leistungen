from __future__ import annotations

from typing import Optional

from utils.commons import result_path, iqr_bounds

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from pandas import DataFrame

from scipy import stats


def summarize(df: pd.DataFrame, var: str, folder: str=None, shapiro: bool = True, to_print=True) -> pd.Series:
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

    if folder is None:
        folder = f"descriptives/{var}"

    if to_print:
        out.to_csv(result_path(folder) / f"summary.csv")

    return out


def grouped_summary(
    df: pd.DataFrame,
    var: str,
    group_col: str,
    folder: str | None = None,
    shapiro: bool = True,
) -> DataFrame:

    if var not in df.columns:
        raise KeyError(f"Variable '{var}' nicht in df.")
    if group_col not in df.columns:
        raise KeyError(f"Gruppenspalte '{group_col}' nicht in df.")

    # Deskriptives (+ optional Shapiro) pro Gruppe
    series_by_group = (
        df.groupby(group_col, dropna=False)[var]
          .apply(lambda s: summarize(s.to_frame(name=var), var=var, shapiro=shapiro, to_print=False))
    )

    out = series_by_group.unstack(level=group_col)
    out.index.name = "Strukturvariable"

    if folder is None:
        folder = f"descriptives/{var}"

    out.to_csv(result_path(folder) / f"summary_{group_col}.csv", index=True)
    return out


# Outlier
def find_outlier(
    df: pd.DataFrame,
    var: str,
    folder: str=None,
    id_col: Optional[str] = "Name",
    iqr_k: float = 1.5,
    to_print = True
) -> DataFrame:

    x = df[var]
    lower, upper = iqr_bounds(x, k=iqr_k)

    mask_lower = df[var] < lower
    mask_upper = df[var] > upper

    cols = []
    if id_col and id_col in df.columns:
        cols.append(id_col)
    cols.append(var)

    out = df.loc[mask_lower | mask_upper, cols].copy()
    if to_print and len(out) > 0:
        out["Outlier-Typ"] = np.where(
            mask_lower.loc[out.index],
            "unterer Outlier",
            "oberer Outlier",
        )

        out["untere Grenze"] = lower
        out["obere Grenze"] = upper

        out.sort_values(["Outlier-Typ", var]).reset_index(drop=True)

        if folder is None:
            folder = f"descriptives/{var}"

        out.to_csv(
            result_path(folder) / "outlier.csv",
            index=False
        )
    return out


def qq_plot(df, var, folder: str=None):
    plt.figure()
    stats.probplot(df[var].to_numpy(), dist="norm", plot=plt)
    plt.title("Q–Q-Plot der 35a-Daten")

    if folder is None:
        folder = f"descriptives/{var}"

    plt.savefig(
        result_path(folder) / f"q_q_plot.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()



# Dist-Plot mit KDE
def plot_distribution(
    df: pd.DataFrame,
    var: str,
    folder: str=None,
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

    if folder is None:
        folder = f"descriptives/{var}"

    plt.savefig(
        result_path(folder) / f"dist.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

# Boxplot nach Gruppen
def boxplot_by_group(
    df: pd.DataFrame,
    var: str,
    group_col: str,
    folder: str=None,
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

    if folder is None:
        folder = f"descriptives/{var}"


    plt.savefig(
        result_path(folder) / f"boxplot_by_{group_col}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

# Scatterplot
def plot_scatter(
    df: pd.DataFrame,
    x: str,
    y: str="35a Hilfen pro 10000",
    folder: str = None,
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

    if folder is None:
        folder = f"descriptives/{x}"

    path = result_path(folder)
    if hue is not None:
        plt.savefig(
            path / f"{y}_scatter_by_{hue}.png",
            dpi=300,
            bbox_inches="tight"
        )
    else:
        plt.savefig(
            path / f"{y}_scatter.png",
            dpi=300,
            bbox_inches="tight"
        )
    plt.close()


# Choroplethenkarte
def show_map(df,
             var,
             group_col: str=None,
             folder: str=None):
    nrw = gpd.read_file("data/NRW-map/dvg1krs_nw.shp")

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

    if folder is None:
        folder = f"descriptives/{var}"

    path = result_path(folder)

    if group_col is None:
        plt.savefig(
            path / f"nrw_map.png",
            dpi=300,
            bbox_inches="tight"
        )
    else:
        plt.savefig(
            path / f"nrw_map_by_{group_col}.png",
            dpi=300,
            bbox_inches="tight"
        )
    plt.close()

# Korrelation
def corr_pair(
    df: pd.DataFrame,
    x: str,
    y: str="35a Hilfen pro 10000",
    folder: str=None,
    method: str="spearman",
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
            "r": np.nan, "p": np.nan,
        }])

    if method == "spearman":
        r = stats.spearmanr(x2, y2)
    elif method == "kendall":
        r = stats.kendalltau(x2, y2)
    else:
        r = stats.pearsonr(x2, y2)



    out= pd.DataFrame([{
        "r": float(r.statistic), "p": float(r.pvalue),

    }])

    if folder is None:
        folder = f"descriptives/{x}"

    if to_print:
        out.to_csv(result_path(folder) / f"{y}_{method}_correlation.csv")
        return out
    else:
        return out



def corr_pair_by_type(
    df: pd.DataFrame,
    x: str,
    group: str,
    y: str="35a Hilfen pro 10000",
    folder: str=None,
) -> DataFrame | None:

    if x == y:
        return None
    out = (df
        .groupby(group)
        .apply(lambda g: corr_pair(g, x, y, to_print=False), include_groups=False)
        .reset_index(level=0)
    )

    if folder is None:
        folder = f"descriptives/{x}"

    out.to_csv(result_path(folder) / f"{y}_correlation_by_{group}.csv")
    return out


def basic_descriptive_analysis(df, var, y="35a Hilfen pro 10000", folder: str=None, shapiro=True) -> None:

    summarize(df, var, folder=folder, shapiro=shapiro),
    plot_distribution(df, var, folder=folder)

    corr_pair(df, var, y, folder=folder)
    plot_scatter(df, var, y, folder=folder)

    show_map(df, var, folder=folder)


def descriptive_analysis_by_group(df, var,  group_col, y="35a Hilfen pro 10000", folder: str=None, shapiro=True)-> None:

    grouped_summary(df, var, group_col, folder=folder, shapiro=shapiro)

    corr_pair_by_type(df, var, group_col, y, folder=folder)

    boxplot_by_group(df, var, group_col, folder=folder)

    plot_scatter(df, var, y, folder=folder, hue=group_col)




