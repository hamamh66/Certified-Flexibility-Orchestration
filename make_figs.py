"""Regenerate quantitative manuscript figures from the replicated campaign."""
import os, sys, pickle
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FIG = os.path.join(HERE, "Figures")

plt.rcParams.update({
    "font.family": "DejaVu Serif", "font.size": 15, "axes.titlesize": 17,
    "axes.labelsize": 16, "xtick.labelsize": 13.5, "ytick.labelsize": 13.5,
    "legend.fontsize": 13, "axes.linewidth": 1.0, "figure.dpi": 300,
    "savefig.dpi": 400, "axes.grid": True, "grid.alpha": 0.28,
    "grid.linewidth": 0.7, "axes.axisbelow": True, "legend.frameon": True,
    "legend.framealpha": 0.93, "legend.edgecolor": "0.8",
    "mathtext.fontset": "dejavuserif"})
NAVY = "#123A5E"; STEEL = "#1B7FB8"; AMBER = "#C8871A"; TEAL = "#2E8B74"
RED = "#B3453A"; GREY = "#9AA1A9"; INK = "#22262B"


def tidy(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("0.35")
    ax.tick_params(colors="0.25", length=4)


def load(tag):
    parts = [pd.read_csv(os.path.join(OUT, f"{tag}_{h}.csv"))
             for h in (0, 1) if os.path.exists(os.path.join(OUT, f"{tag}_{h}.csv"))]
    return pd.concat(parts, ignore_index=True)


def mci(x, level=0.95):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    m = x.mean()
    if len(x) < 2:
        return m, 0.0
    h = st.t.ppf(0.5 + level / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, h


CTL = ["B1", "B2", "B3", "B4", "B5", "B6", "P"]
SUB = ["uncontrolled", "ToU rule", "carbon rule", "MPC perfect",
       "MPC realistic", "robust MPC", "proposed"]
COLS = [GREY, GREY, GREY, TEAL, AMBER, AMBER, NAVY]


def fig4():
    df = load("caseA")
    panels = [("uncond_service_rate_pct", "(a) Unconditional service rate (%)", None),
              ("cond_delivery_rate_pct", "(b) Delivery rate among accepted (%)", None),
              ("false_activation_pct", "(c) False activation (%)", None),
              ("net_service_benefit", "(d) Net service benefit (k\\$/yr)", 0.0)]
    fig, axes = plt.subplots(1, 4, figsize=(21, 5.8))
    for ax, (m, title, ref) in zip(axes, panels):
        means, errs = [], []
        for c in CTL:
            v = df[df.controller == c][m].values
            if m == "net_service_benefit":
                v = v / 1e3
            mu, h = mci(v)
            means.append(mu); errs.append(h)
        b = ax.bar(range(7), means, yerr=errs, capsize=4,
                   color=COLS, edgecolor="white", linewidth=1.1, width=0.72,
                   error_kw=dict(ecolor=INK, lw=1.4))
        b[6].set_edgecolor(INK); b[6].set_linewidth(1.8)
        if ref is not None:
            ax.axhline(ref, ls="--", lw=1.3, color="0.45", zorder=1)
        ax.set_title(title, pad=12, color=INK)
        ax.set_xticks(range(7))
        ax.set_xticklabels([f"{c}\n{s}" for c, s in zip(CTL, SUB)],
                           rotation=45, ha="right", fontsize=11.5)
        ax.grid(axis="x", visible=False)
        tidy(ax)
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_reliability.png"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig9():
    df = load("frontier")
    fam = {"B6": [("B6r0.00", "0"), ("B6r0.05", "0.05"), ("B6r0.10", "0.10"),
                  ("B6r0.20", "0.20"), ("B6r0.30", "0.30"), ("B6r0.40", "0.40")],
           "P": [("Pa0.30", "0.30"), ("Pa0.20", "0.20"), ("Pa0.10", "0.10"),
                 ("Pa0.05", "0.05"), ("Pa0.02", "0.02")]}
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    for ax, ym, ylab in [(axes[0], "uncond_service_rate_pct",
                          "Unconditional service rate (%)"),
                         (axes[1], "net_service_benefit",
                          "Net service benefit (k\\$/yr)")]:
        for name, col, lbl in [("B6", AMBER, "robust MPC (margin $r$)"),
                               ("P", NAVY, "proposed (nominal $\\alpha$)")]:
            xs, ys, xe, ye, ann = [], [], [], [], []
            for tag, a in fam[name]:
                g = df[df.controller == tag]
                x, hx = mci(g["EV_shortfall_kWh"].values / 1e3)
                v = g[ym].values / (1e3 if ym == "net_service_benefit" else 1)
                y, hy = mci(v)
                xs.append(x); ys.append(y); xe.append(hx); ye.append(hy)
                ann.append(a)
            ax.errorbar(xs, ys, xerr=xe, yerr=ye, fmt="-o", color=col,
                        ms=9, lw=2.2, capsize=3.5, label=lbl)
            for x, y, a in zip(xs, ys, ann):
                ax.annotate(a, (x, y), textcoords="offset points",
                            xytext=(8, 7), fontsize=12.5, color=col)
        ax.set_xlabel("Unmet EV departure energy (MWh/yr)")
        ax.set_ylabel(ylab)
        ax.legend(loc="lower right")
        tidy(ax)
    axes[0].set_title("(a) Service reliability vs mobility cost", pad=12)
    axes[1].set_title("(b) Settlement value vs mobility cost", pad=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig9_frontier.png"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig5():
    agg = pd.read_csv(os.path.join(OUT, "twostage_agg.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    ax = axes[0]
    ax.plot([0.6, 1.0], [0.6, 1.0], "--", color="0.5", lw=1.3,
            label="ideal calibration")
    ax.errorbar(agg.nominal, agg.cov_one, yerr=agg.cov_one_sd, fmt="-s",
                color=STEEL, ms=9, lw=2.2, capsize=3.5,
                label="single-stage (magnitude only)")
    ax.errorbar(agg.nominal, agg.cov_two, yerr=agg.cov_two_sd, fmt="-o",
                color=NAVY, ms=10, lw=2.4, capsize=3.5,
                label="two-stage (availability then magnitude)")
    ax.set_xlabel("Nominal confidence $1-\\alpha$")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("(a) Coverage vs nominal level (locked calibration)", pad=12)
    ax.legend(loc="upper left")
    tidy(ax)
    ax = axes[1]
    ax.plot(agg.cov_one, agg.mean_cert_one, "-s", color=STEEL, ms=9, lw=2.2,
            label="single-stage")
    ax.plot(agg.cov_two, agg.mean_cert_two, "-o", color=NAVY, ms=10, lw=2.4,
            label="two-stage")
    for _, r in agg.iterrows():
        ax.annotate(f"{r.nominal:.2f}", (r.cov_one, r.mean_cert_one),
                    textcoords="offset points", xytext=(7, 6),
                    fontsize=12, color=STEEL)
        if r.mean_cert_two > 0:
            ax.annotate(f"{r.nominal:.2f}", (r.cov_two, r.mean_cert_two),
                        textcoords="offset points", xytext=(7, -14),
                        fontsize=12, color=NAVY)
    ax.set_xlabel("Empirical coverage")
    ax.set_ylabel("Mean certified magnitude (kW)")
    ax.set_title("(b) Coverage--value curve", pad=12)
    ax.legend(loc="upper right")
    tidy(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_coverage_value.png"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig6():
    rows = []
    for h in (0, 1):
        p = os.path.join(OUT, f"caseA_{h}_records.pkl")
        with open(p, "rb") as fh:
            kept = pickle.load(fh)
        for seed, recs in kept.items():
            d = recs["P"]["deliv"]
            rows.append(d[d.promised > 1e-6][["promised", "delivered"]])
    d = pd.concat(rows, ignore_index=True)
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    hb = ax.hexbin(d.promised, d.delivered, gridsize=34, cmap="Blues",
                   mincnt=1, linewidths=0.2, edgecolors="white")
    lim = max(d.promised.max(), d.delivered.max()) * 1.04
    ax.plot([0, lim], [0, lim], "--", color=RED, lw=1.6,
            label="delivered = promised")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Promised reduction (kW)")
    ax.set_ylabel("Delivered reduction vs common baseline (kW)")
    cb = fig.colorbar(hb, ax=ax, pad=0.015)
    cb.set_label("Accepted requests per cell", fontsize=13.5)
    ax.legend(loc="upper left")
    tidy(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6_promise_delivery.png"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig7():
    df = load("ablation")
    labs = [("full", "full method"), ("nocert", "$-$ de-rating layer"),
            ("norisk", "$-$ risk term"), ("noinfo", "$-$ information term"),
            ("nobins", "$-$ regime bins")]
    panels = [("uncond_service_rate_pct", "(a) Unconditional service rate (%)"),
              ("false_activation_pct", "(b) False activation (%)"),
              ("EV_shortfall_kWh", "(c) EV shortfall (MWh/yr)"),
              ("net_service_benefit", "(d) Net service benefit (k\\$/yr)")]
    fig, axes = plt.subplots(1, 4, figsize=(21, 5.4))
    for ax, (m, title) in zip(axes, panels):
        means, errs = [], []
        for tag, _ in labs:
            v = df[df.controller == tag][m].values
            if m in ("net_service_benefit", "EV_shortfall_kWh"):
                v = v / 1e3
            mu, h = mci(v)
            means.append(mu); errs.append(h)
        cols = [NAVY, STEEL, STEEL, STEEL, STEEL]
        b = ax.bar(range(5), means, yerr=errs, capsize=4, color=cols,
                   edgecolor="white", linewidth=1.1, width=0.7,
                   error_kw=dict(ecolor=INK, lw=1.4))
        b[0].set_edgecolor(INK); b[0].set_linewidth(1.8)
        ax.set_title(title, pad=12)
        ax.set_xticks(range(5))
        ax.set_xticklabels([l for _, l in labs], rotation=45, ha="right",
                           fontsize=11.5)
        ax.grid(axis="x", visible=False)
        tidy(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig7_ablation.png"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig8():
    frames = []
    for sev in (0.0, 0.5, 1.0, 2.0, 3.0):
        p = os.path.join(OUT, f"stress_{sev}.csv")
        if os.path.exists(p):
            d = pd.read_csv(p); d["sev"] = sev; frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    panels = [("uncond_service_rate_pct", "(a) Unconditional service rate (%)"),
              ("false_activation_pct", "(b) False activation (%)"),
              ("EV_shortfall_kWh", "(c) EV shortfall (MWh/yr)")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    for ax, (m, title) in zip(axes, panels):
        for c, col, lbl in [("B5", AMBER, "B5 MPC realistic"),
                            ("P", NAVY, "proposed")]:
            xs, ys, es = [], [], []
            for sev in sorted(df.sev.unique()):
                v = df[(df.controller == c) & (df.sev == sev)][m].values
                if m == "EV_shortfall_kWh":
                    v = v / 1e3
                mu, h = mci(v)
                xs.append(sev); ys.append(mu); es.append(h)
            ys, es = np.array(ys), np.array(es)
            ax.plot(xs, ys, "-o", color=col, ms=9, lw=2.2, label=lbl)
            ax.fill_between(xs, ys - es, ys + es, color=col, alpha=0.18, lw=0)
        ax.set_xlabel("Forecast-error severity (\\times nominal)")
        ax.set_title(title, pad=12)
        ax.legend(loc="best")
        tidy(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig8_stress.png"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    for f in sys.argv[1:] or ["fig4", "fig5", "fig6", "fig7", "fig8", "fig9"]:
        globals()[f]()
        print(f, "done", flush=True)
