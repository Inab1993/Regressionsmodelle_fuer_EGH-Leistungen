from __future__ import annotations

import math
from typing import Optional

from matplotlib.figure import Figure
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

from utils.commons import result_path, iqr_bounds

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from pandas import DataFrame

from scipy import stats


def summarize(df: pd.DataFrame, var: str) -> pd.Series:
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

    out = pd.Series(summary, name=f"{var}").round(3)
    out.index.name = "Variable"
    return out


def grouped_summary(
    df: pd.DataFrame,
    var: str,
    group_col: str,
) -> DataFrame:

    if var not in df.columns:
        raise KeyError(f"Variable '{var}' nicht in df.")
    if group_col not in df.columns:
        raise KeyError(f"Gruppenspalte '{group_col}' nicht in df.")

    series_by_group = (
        df.groupby(group_col, dropna=False)[var]
          .apply(lambda s: summarize(s.to_frame(name=var), var=var))
    )

    out = series_by_group.unstack(level=group_col)
    out.index.name = "Variable"
    return out


def find_outlier(
    df: pd.DataFrame,
    var: str,
    id_col: Optional[str] = "Name",
    iqr_k: float = 1.5,
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
    if len(out) > 0:
        out["Outlier-Typ"] = np.where(
            mask_lower.loc[out.index],
            "unterer Outlier",
            "oberer Outlier",
        )

        out["untere Grenze"] = round(lower, 2)
        out["obere Grenze"] = round(upper, 2)

        out.sort_values(["Outlier-Typ", var]).reset_index(drop=True)

    return out


def plot(
    df: pd.DataFrame,
    var: str,
    type: str,
    group_col: str | None = None,
    title: str | None = None,
    bins: int = 20,
    bw_method: float = 0.5,
) -> Figure | None:

    def _dist_kde(ax, dataframe: pd.DataFrame, variable: str) -> None:
        x = dataframe[variable].dropna()
        if x.size < 3:
            return

        vals = x.to_numpy()

        ax.hist(vals, bins=bins, density=True)
        ax.set_xlabel(variable)
        ax.set_ylabel("Dichte")

        # KDE nur wenn Spannweite > 0 (sonst Fehler/Quatsch)
        if vals.min() == vals.max():
            return

        xx = np.linspace(vals.min(), vals.max(), 200)
        kde = stats.gaussian_kde(vals, bw_method=bw_method)
        ax.plot(xx, kde(xx))

    if var not in df.columns:
        raise KeyError(f"Spalte '{var}' nicht in df.")
    if group_col is not None and group_col not in df.columns:
        raise KeyError(f"Spalte '{group_col}' nicht in df.")

    base_title = title if title is not None else f"{type}-Plot der {var}-Daten"

    if group_col is None:
        fig, ax = plt.subplots()

        if type == "qq":
            vals = df[var].dropna().to_numpy()
            if vals.size < 3:
                plt.close(fig)
                return None
            stats.probplot(vals, dist="norm", plot=ax)

        elif type == "dist_kde":
            _dist_kde(ax, df, var)

        else:
            plt.close(fig)
            raise ValueError(f"Unbekannter type: {type}")

        ax.set_title(base_title)
        fig.tight_layout()
        return fig

    # --- mit Gruppierung ---
    groups = list(df.groupby(group_col, dropna=False))
    n_groups = len(groups)
    if n_groups == 0:
        return None

    ncols = math.ceil(math.sqrt(n_groups))
    nrows = math.ceil(n_groups / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    valid_plot = False

    for ax, (g_value, g_df) in zip(axes, groups):
        vals = g_df[var].dropna().to_numpy()
        if vals.size < 3:
            ax.set_visible(False)
            continue

        if type == "qq":
            stats.probplot(vals, dist="norm", plot=ax)

        elif type == "dist_kde":
            _dist_kde(ax, g_df, var)

        else:
            ax.set_visible(False)
            continue

        ax.set_title(f"{group_col}: {g_value}")
        valid_plot = True

    for ax in axes[len(groups):]:
        ax.set_visible(False)

    if not valid_plot:
        plt.close(fig)
        return None

    fig.suptitle(base_title)
    fig.tight_layout()
    return fig



# Boxplot nach Gruppen
def boxplot_by_group(
    df: pd.DataFrame,
    var: str,
    group_col: str,
    title: str=None,
    showfliers: bool=False
) -> Figure | None:
    plot_df = df[[group_col, var]].dropna()
    if plot_df.empty:
        return None
    labels = plot_df[group_col].unique()

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(labels)), 6))

    plot_title = title if title is not None else f"boxplot_{var}_nach_{group_col}"
    ax.set_title(plot_title)

    sns.boxplot(
        data=plot_df,
        x=group_col,
        y=var,
        showfliers=showfliers,
        ax=ax
    )

    sns.stripplot(
        data=plot_df,
        x=group_col,
        y=var,
        color="black",
        alpha=0.5,
        jitter=True,
        ax=ax
    )

    ax.set_ylabel(var)
    ax.set_xlabel(group_col)
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    return fig

# Scatterplot
def plot_scatter(
    df: pd.DataFrame,
    x: str,
    y: str="35a Hilfen pro 10000",
    hue: Optional[str] = None,
) -> Figure | None:
    if x == y:
        return None
    d = df.copy()


    fig, ax = plt.subplots()
    plt.figure()
    if hue is None:
        ax.scatter(d[x], d[y])
    else:
        for key, g in d.groupby(hue, dropna=False):
            ax.scatter(g[x], g[y], label=str(key))
        ax.legend(title=hue)

    ax.set_title(f"Scatterplot von {x} und {y}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    fig.tight_layout()

    return fig


# Choroplethenkarte
def show_map(df,
             var,
             group_col: str=None):
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
    nrw.plot(column=var, ax=ax, legend=True, cmap="OrRd", edgecolor="black",  legend_kwds={"shrink": 0.5})

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

    return fig

# Korrelation
def corr_pair(
    df: pd.DataFrame,
    x: str,
    y: str="35a Hilfen pro 10000",
    method: str="spearman",
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


    return out



def corr_pair_by_type(
    df: pd.DataFrame,
    x: str,
    group: str,
    y: str="35a Hilfen pro 10000",
    method: str="spearman",
) -> DataFrame | None:

    if x == y:
        return None
    out = (df
        .groupby(group)
        .apply(lambda g: corr_pair(g, x, y,method=method), include_groups=False)
        .reset_index(level=0)
    )
    return out


def vif(df, vars: tuple):
    X = df[[*vars]]
    X = sm.add_constant(X)

    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i)
        for i in range(X.shape[1])
    ]

    vif_data = vif_data[vif_data["Variable"] != "const"]
    vif_data = vif_data.sort_values("VIF", ascending=False)
    return vif_data



def correlations(df: pd.DataFrame, vars: tuple, method="spearman"):
    corr = df[[*vars]].corr(method=method)

    # Nur obere Dreiecksmatrix (ohne Diagonale)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    corr_pairs = corr.where(mask)

    corr_pairs = (
        corr_pairs
        .stack()
        .reset_index()
    )
    corr_pairs.columns = ["Variable_1", "Variable_2", "r"]

    filtered = corr_pairs[abs(corr_pairs["r"]) > 0.5]

    filtered.sort_values("r", ascending=False)
    return filtered

def chi2_test_report(df, var1, var2):
    table = pd.crosstab(df[var1], df[var2])
    chi2, p, dof, expected = chi2_contingency(table)
    n = table.to_numpy().sum()
    r, k = table.shape
    cramers_v = np.sqrt(chi2 / (n * min(r - 1, k - 1)))

    expected_df = pd.DataFrame(
        expected,
        index=table.index,
        columns=table.columns
    )

    report = (
        f"Chi-Quadrat-Statistik: {chi2:.4f}\n"
        f"p-Wert: {p:.4f}\n"
        f"Freiheitsgrade: {dof}\n"
        f"Stichprobengröße n: {n}\n"
        f"Cramérs V: {cramers_v:.4f}\n"
        f"Minimale erwartete Häufigkeit: {expected.min():.4f}\n\n"
        f"Erwartete Häufigkeiten:\n{expected_df.to_string()}\n"
    )
    return report


def describe_and_save(df, var, y="35a Hilfen pro 10000", folder: str=None, map: str = None) -> None:
    folder = folder if folder else f"descriptive/{var}"
    summarize(df, var).to_csv(result_path(folder) / f"summary.csv")
    fig = plot(df, var, type="dist_kde")
    fig.savefig(result_path(folder) / f"dist_{var}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    if var != y:
        corr_pair(df, var, y).to_csv(result_path(folder) / f"corr_pair_{y}_to_{var}.csv", index=False)
    if map == "NRW":
        fig = show_map(df, var)
        fig.savefig(result_path(folder) / f"map_{var}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def describe_and_save_grouped(df, var,  group_col, y="35a Hilfen pro 10000", folder: str=None)-> None:
    folder = folder if folder else f"descriptive/{var}"
    grouped_summary(df, var, group_col).to_csv(result_path(folder) / f"summary_{group_col}.csv")
    if var != y:
        corr_pair_by_type(df, var, group_col, y).to_csv(result_path(folder) / f"corr_{var}_to_{y}_by_{group_col}_spearman.csv")
        fig = plot_scatter(df, var, y, hue=group_col)
        fig.savefig(result_path(folder) / f"scatter_{var}_to_{y}_by_{group_col}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    fig=boxplot_by_group(df, var, group_col)
    fig.savefig(result_path(folder) / f"boxplot_{var}_by_{group_col}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)






