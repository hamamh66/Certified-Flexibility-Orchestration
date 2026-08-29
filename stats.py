"""Paired statistical analysis per the pre-specified protocol.

Primary comparisons (pre-registered in the submitted manuscript, Sec. 3.7):
  (i)  P vs B5: uncond service rate, false activation, EV shortfall,
       net service benefit;
  (ii) P vs B6: EV shortfall, comfort violations.
Secondary family (Holm-Bonferroni at 5%): carbon, cost, peak, comfort,
battery cycling for P vs B5.
Equivalence (TOST) margins, pre-specified: 1% carbon, 1% cost, 5% comfort
violation-hours (relative to the B5 mean).
"""
import sys, os, itertools, json
import numpy as np, pandas as pd
from scipy import stats as st

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
rng = np.random.default_rng(7)


def load(tag):
    parts = [pd.read_csv(os.path.join(OUT, f"{tag}_{h}.csv")) for h in (0, 1)
             if os.path.exists(os.path.join(OUT, f"{tag}_{h}.csv"))]
    return pd.concat(parts, ignore_index=True)


def bca_ci(x, B=10000, level=0.95, stat=np.mean):
    """BCa bootstrap CI."""
    x = np.asarray(x, float)
    n = len(x)
    th = stat(x)
    boots = np.array([stat(rng.choice(x, n, replace=True)) for _ in range(B)])
    z0 = st.norm.ppf(max(min((boots < th).mean(), 1 - 1e-9), 1e-9))
    jack = np.array([stat(np.delete(x, i)) for i in range(n)])
    jm = jack.mean()
    num = ((jm - jack) ** 3).sum()
    den = 6.0 * (((jm - jack) ** 2).sum() ** 1.5)
    a = num / den if den > 0 else 0.0
    al = (1 - level) / 2
    lo_q = st.norm.cdf(z0 + (z0 + st.norm.ppf(al)) / (1 - a * (z0 + st.norm.ppf(al))))
    hi_q = st.norm.cdf(z0 + (z0 + st.norm.ppf(1 - al)) / (1 - a * (z0 + st.norm.ppf(1 - al))))
    return th, float(np.quantile(boots, lo_q)), float(np.quantile(boots, hi_q))


def paired(df, m, c1, c2):
    a = df[df.controller == c1].set_index("seed")[m].sort_index()
    b = df[df.controller == c2].set_index("seed")[m].sort_index()
    common = a.index.intersection(b.index)
    d = (a.loc[common] - b.loc[common]).values
    d = d[~np.isnan(d)]
    return d


def wilcoxon_row(df, m, c1, c2):
    d = paired(df, m, c1, c2)
    out = dict(metric=m, pair=f"{c1}-{c2}", n=len(d), mean_diff=float(np.mean(d)))
    _, lo, hi = bca_ci(d)
    out["ci_lo"], out["ci_hi"] = lo, hi
    if np.allclose(d, 0):
        out["p_wilcoxon"] = 1.0
        out["p_ttest"] = 1.0
        out["rank_biserial"] = 0.0
        out["cohen_dz"] = 0.0
        return out
    w = st.wilcoxon(d, zero_method="wilcox", method="auto")
    out["p_wilcoxon"] = float(w.pvalue)
    out["p_ttest"] = float(st.ttest_rel(d, np.zeros_like(d)).pvalue)
    nz = d[d != 0]
    n = len(nz)
    r_plus = st.rankdata(np.abs(nz))[nz > 0].sum()
    tot = n * (n + 1) / 2
    out["rank_biserial"] = float(2 * r_plus / tot - 1)
    out["cohen_dz"] = float(np.mean(d) / (np.std(d, ddof=1) + 1e-12))
    out["median_diff"] = float(np.median(d))
    return out


def tost(df, m, c1, c2, margin):
    """Two one-sided paired t-tests against +-margin (absolute units)."""
    d = paired(df, m, c1, c2)
    n = len(d)
    se = np.std(d, ddof=1) / np.sqrt(n) + 1e-12
    t_lo = (np.mean(d) + margin) / se     # H0: diff <= -margin
    t_hi = (np.mean(d) - margin) / se     # H0: diff >= +margin
    p_lo = 1 - st.t.cdf(t_lo, n - 1)
    p_hi = st.t.cdf(t_hi, n - 1)
    p = max(p_lo, p_hi)
    return dict(metric=m, pair=f"{c1}-{c2}", margin=margin, n=n,
                mean_diff=float(np.mean(d)), p_tost=float(p),
                equivalent=bool(p < 0.05))


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    mx = 0.0
    for rank, i in enumerate(order):
        mx = max(mx, (m - rank) * pvals[i])
        adj[i] = min(mx, 1.0)
    return adj


def analyse(tag="caseA", label=""):
    df = load(tag)
    # ---- descriptive table: mean [CI] per controller per metric
    metrics = ["CO2_t_per_yr", "cost_per_yr", "peak_kW", "comfort_viol_h",
               "EV_shortfall_kWh", "bat_cycles", "acceptance_rate_pct",
               "abstention_rate_pct", "cond_delivery_rate_pct",
               "uncond_service_rate_pct", "delivered_frac_of_requested",
               "delivered_frac_of_accepted", "shortfall_MWh",
               "overdelivery_MWh", "false_activation_pct",
               "net_service_benefit", "requested_MWh", "accepted_MWh",
               "delivered_MWh", "service_revenue_yr", "service_penalty_yr"]
    desc = []
    for c in df.controller.unique():
        for m in metrics:
            x = df[df.controller == c][m].dropna().values
            if len(x) == 0:
                continue
            mu, lo, hi = bca_ci(x, B=4000)
            desc.append(dict(controller=c, metric=m, mean=mu, ci_lo=lo,
                             ci_hi=hi, n=len(x)))
    pd.DataFrame(desc).to_csv(os.path.join(OUT, f"desc_{tag}.csv"), index=False)

    # ---- primary + secondary tests
    primary = [("uncond_service_rate_pct", "P", "B5"),
               ("false_activation_pct", "P", "B5"),
               ("EV_shortfall_kWh", "P", "B5"),
               ("net_service_benefit", "P", "B5"),
               ("EV_shortfall_kWh", "P", "B6"),
               ("comfort_viol_h", "P", "B6")]
    secondary = [("CO2_t_per_yr", "P", "B5"), ("cost_per_yr", "P", "B5"),
                 ("peak_kW", "P", "B5"), ("comfort_viol_h", "P", "B5"),
                 ("bat_cycles", "P", "B5")]
    rows_p = [wilcoxon_row(df, m, a, b) for m, a, b in primary]
    rows_s = [wilcoxon_row(df, m, a, b) for m, a, b in secondary]
    for grp in (rows_p, rows_s):
        adj = holm([r["p_wilcoxon"] for r in grp])
        for r, a in zip(grp, adj):
            r["p_holm"] = float(a)
    tests = pd.DataFrame(rows_p + rows_s)
    tests["family"] = ["primary"] * len(rows_p) + ["secondary"] * len(rows_s)
    tests.to_csv(os.path.join(OUT, f"tests_{tag}.csv"), index=False)

    # ---- TOST equivalence, margins pre-specified relative to B5 mean
    b5 = df[df.controller == "B5"]
    eq = [tost(df, "CO2_t_per_yr", "P", "B5", 0.01 * b5.CO2_t_per_yr.mean()),
          tost(df, "cost_per_yr", "P", "B5", 0.01 * b5.cost_per_yr.mean()),
          tost(df, "comfort_viol_h", "P", "B5",
               0.05 * max(b5.comfort_viol_h.mean(), 1.0)),
          tost(df, "bat_cycles", "P", "B5", 0.05 * b5.bat_cycles.mean())]
    pd.DataFrame(eq).to_csv(os.path.join(OUT, f"tost_{tag}.csv"), index=False)
    print(f"analysed {tag}: {len(df)} rows, {df.seed.nunique()} seeds")
    return df


if __name__ == "__main__":
    for tag in sys.argv[1:] or ["caseA"]:
        analyse(tag)
