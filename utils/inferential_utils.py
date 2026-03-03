from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
import statsmodels.api as sm

from utils.commons import result_path


def scatter_with_ols_line(df: pd.DataFrame, x: str, y: str, group: str | None = None) -> None:
    d = df.copy()
    d = d[[x, y] + ([group] if group else [])].dropna()

    plt.figure()
    if group and group in d.columns:
        for g, sub in d.groupby(group):
            plt.scatter(sub[x], sub[y], label=str(g), alpha=0.8)
    else:
        plt.scatter(d[x], d[y], alpha=0.85)

    X = sm.add_constant(d[x].to_numpy())
    model = sm.OLS(d[y].to_numpy(), X).fit()
    xs = np.linspace(d[x].min(), d[x].max(), 200)
    ys = model.params[0] + model.params[1] * xs
    plt.plot(xs, ys)

    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"Scatter + OLS-Linie: {y} ~ {x}")
    if group and group in d.columns:
        plt.legend(title=group)
    plt.tight_layout()
    plt.show()


def hetero_diagnostics(model, var) -> None:
    fitted = model.fittedvalues
    resid = model.resid

    plt.figure()
    plt.scatter(fitted, resid, alpha=0.85)
    plt.axhline(0)
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Fitted")
    plt.tight_layout()
    plt.show()

    # model.model ist das ursprüngliche OLS Modellobjekt, exog die Matrix der unabh. Variable (im Gegensatz zu endog)
    exog = model.model.exog

    # beide Tests geringe Power bei kleinen Stichproben (n=30)
    bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(resid, exog)
    w_lm, w_lm_p, w_f, w_f_p = het_white(resid, exog)

    df_out = pd.DataFrame({
        "Test": ["Breusch-Pagan", "White"],
        "LM-Statistik": [bp_lm, w_lm],
        "LM p-Value": [bp_lm_p, w_lm_p],
        "F-Statistik": [bp_f, w_f],
        "F p-Value": [bp_f_p, w_f_p],
    })  # Interpretation: p-Values über 0.05 = Homoskedastizität
        # LM = n * R², aus dem Vergleich mit Chi² entsteht der p Wert
        # F = Fisher-Verteilung, besser bei kleineren n
        # White Test ist weniger sensitiv, wenn Normalverteilungsannahmen verletzt werden

    path = Path(f"../../tables/inferential")
    path.mkdir(parents=True, exist_ok=True)

    df_out = df_out.round(5)
    df_out.to_csv(path / f"{var}_het_tests.csv", index=False)

    print(df_out)

def shapiro_wilk(df, var: str, group_col: str = None):
    if var not in df.columns:
        raise KeyError(f"Variable '{var}' nicht in DataFrame.")

    if group_col is None:
        x = df[var].dropna()

        sh = stats.shapiro(x)
        summary = {
            "Shapiro-Statistik": float(sh.statistic),
            "p-Value": float(sh.pvalue),
        }

        out = pd.Series(summary, name=var).round(3)
        out.index.name = "Kennwert"
        return out

    if group_col not in df.columns:
        raise KeyError(f"Gruppenvariable '{group_col}' nicht in DataFrame.")

    results = []

    for group, sub in df.groupby(group_col):
        x = sub[var].dropna()

        if len(x) < 3:
            results.append({
                "Gruppe": group,
                "Shapiro-Statistik": None,
                "p-Value": None,
            })
            continue

        sh = stats.shapiro(x)

        results.append({
            "Gruppe": group,
            "Shapiro-Statistik": float(sh.statistic),
            "p-Value": float(sh.pvalue),
        })

    out = pd.DataFrame(results).set_index("Gruppe").round(3)
    return out

def run_regression(df_sub, x: str, y:str):
    d = df_sub[["Name",x, y]].copy()
    d = d[[x, y]].dropna()
    X = sm.add_constant(d[x])
    model = sm.OLS(d[y], X).fit()

    print("\n--- OLS-results ---")
    model.summary()
    print(f"n = {len(d)}")
    print(f"R² = {model.rsquared:.3f}")
    print(f"b = {model.params[x]:.3f}")
    print(f"p = {model.pvalues[x]:.4f}")

    d["y_hat"] = model.predict(X)
    d["residual"] = d[y] - d["y_hat"]

    return model, d



def robust_cov(model):
    robust = model.get_robustcov_results(cov_type="HC3", use_t=True)   #use_t=true setzt t-verteilung vorraus, gut bei geringer stichprobengröße und normalverteilungsähnlich, aber tendenziell mehr ausreißern

    print("\nOLS mit robusten SE (HC3):")
    print(f"n = {int(robust.nobs)}")
    print(f"R² = {robust.rsquared:.3f}")
"""    print(f"b = {robust.params[x]:.3f}")
    print(f"R² = {robust.rsquared:.3f}")
    print(f"p = {robust.pvalues[x]:.4f}")
"""


def t_test(df, cat, var):
    d = df[[cat, var]].copy()

    cats = d[cat].unique()

    cat_y = d.loc[d[cat] == cats[0], var].to_numpy()
    cat_x = d.loc[d[cat] == cats[1], var].to_numpy()

    # students-t-Test (für annähernd normalverteilte Variablen)
    t_stat, p_val = stats.ttest_ind(cat_x, cat_y)

    mean_diff = np.mean(cat_x) - np.mean(cat_y)
    s1 = np.var(cat_x, ddof=1)
    s2 = np.var(cat_y, ddof=1)
    n1, n2 = cat_x.size, cat_y.size

    sp = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    cohen_d = mean_diff / sp

    t_test_result = {
        "t-Test": t_stat,
        "p_val": p_val,
        "Effektstärke": cohen_d,
    }

    results = pd.Series(t_test_result,name=var).round(3)

    results.index.name="Variable"
    return results




