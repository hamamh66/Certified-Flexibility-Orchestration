"""Generate LaTeX tables from campaign outputs into Sections/Tab*.tex."""
import os, sys
import numpy as np, pandas as pd
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
SEC = os.path.join(HERE, "Sections")


def load(tag):
    parts = [pd.read_csv(os.path.join(OUT, f"{tag}_{h}.csv"))
             for h in (0, 1) if os.path.exists(os.path.join(OUT, f"{tag}_{h}.csv"))]
    return pd.concat(parts, ignore_index=True)


def mci(x, level=0.95):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan
    m = x.mean()
    if len(x) < 2:
        return m, 0.0
    h = st.t.ppf(0.5 + level / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, h


def cell(x, dec=1, scale=1.0):
    m, h = mci(np.asarray(x, float) * scale)
    if np.isnan(m):
        return "--"
    return f"{m:.{dec}f}\\,$\\pm$\\,{h:.{dec}f}"


ROWS = [
    ("Operational CO$_2$ (t/yr)", "CO2_t_per_yr", 1, 1.0),
    ("Energy cost (k\\$/yr)", "cost_per_yr", 1, 1e-3),
    ("Peak import (kW)", "peak_kW", 0, 1.0),
    ("Comfort violations (h/yr)", "comfort_viol_h", 0, 1.0),
    ("EV shortfall (MWh/yr)", "EV_shortfall_kWh", 1, 1e-3),
    ("Battery equiv.\\ full cycles (/yr)", "bat_cycles", 0, 1.0),
    ("MIDRULE", None, 0, 0),
    ("Acceptance rate (\\%)", "acceptance_rate_pct", 1, 1.0),
    ("Abstention rate (\\%)", "abstention_rate_pct", 1, 1.0),
    ("Requested flexibility (MWh)", "requested_MWh", 2, 1.0),
    ("Accepted flexibility (MWh)", "accepted_MWh", 2, 1.0),
    ("Delivered flexibility (MWh)", "delivered_MWh", 2, 1.0),
    ("Conditional delivery rate (\\%)", "cond_delivery_rate_pct", 1, 1.0),
    ("Unconditional service rate (\\%)", "uncond_service_rate_pct", 1, 1.0),
    ("False activation (\\%)", "false_activation_pct", 1, 1.0),
    ("Shortfall (MWh)", "shortfall_MWh", 2, 1.0),
    ("Over-delivery (MWh)", "overdelivery_MWh", 2, 1.0),
    ("Settlement revenue (\\$/yr)", "service_revenue_yr", 0, 1.0),
    ("Settlement penalties (\\$/yr)", "service_penalty_yr", 0, 1.0),
    ("Net service benefit (\\$/yr)", "net_service_benefit", 0, 1.0),
]


def main_table(tag, fname, caption, label, ctl, heads):
    df = load(tag)
    n = df.seed.nunique()
    lines = [
        "\\begin{table*}[htbp]", "\\centering",
        f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        "\\scriptsize", "\\setlength{\\tabcolsep}{2pt}",
        "\\begin{tabular}{l" + "c" * len(ctl) + "}", "\\toprule",
        "Metric & " + " & ".join(heads) + " \\\\", "\\midrule"]
    for name, m, dec, sc in ROWS:
        if name == "MIDRULE":
            lines.append("\\midrule")
            continue
        cells = [cell(df[df.controller == c][m].values, dec, sc) for c in ctl]
        lines.append(name + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    open(os.path.join(SEC, fname), "w").write("\n".join(lines))
    print(fname, "written, n =", n)


def frontier_table():
    df = load("frontier")
    fam = [("Robust MPC", [("B6r0.00", "$r{=}0.00$"), ("B6r0.05", "$r{=}0.05$"),
                           ("B6r0.10", "$r{=}0.10$"), ("B6r0.20", "$r{=}0.20$"),
                           ("B6r0.30", "$r{=}0.30$"), ("B6r0.40", "$r{=}0.40$")]),
           ("Proposed", [("Pa0.30", "$\\alpha{=}0.30$"), ("Pa0.20", "$\\alpha{=}0.20$"),
                         ("Pa0.10", "$\\alpha{=}0.10$"), ("Pa0.05", "$\\alpha{=}0.05$"),
                         ("Pa0.02", "$\\alpha{=}0.02$")])]
    lines = ["\\begin{table}[htbp]", "\\centering",
             "\\caption{Case A reliability--mobility frontiers under the common "
             "external request: mean $\\pm$ 95\\% CI over 10 paired realisations. "
             "Service rate is the unconditional rate over all requests; the "
             "proposed method's knob moves abstention, not delivered "
             "reliability, which is the operational signature of the "
             "calibration failure analysed in Section~\\ref{sec:certification}.}",
             "\\label{tab:frontier}", "\\footnotesize",
             "\\setlength{\\tabcolsep}{3.5pt}",
             "\\begin{tabular}{llccccc}", "\\toprule",
             "Family & Knob & Service & Abstention & EV shortfall & Comfort & "
             "Net benefit \\\\",
             " & & rate (\\%) & (\\%) & (MWh/yr) & (h/yr) & (\\$/yr) \\\\",
             "\\midrule"]
    for famname, entries in fam:
        lines.append(f"\\multirow{{{len(entries)}}}{{*}}{{{famname}}}")
        for tag, lbl in entries:
            g = df[df.controller == tag]
            row = [lbl,
                   cell(g.uncond_service_rate_pct, 1),
                   cell(g.abstention_rate_pct, 1),
                   cell(g.EV_shortfall_kWh, 1, 1e-3),
                   cell(g.comfort_viol_h, 0),
                   cell(g.net_service_benefit, 0)]
            lines.append(" & " + " & ".join(row) + " \\\\")
        if famname == "Robust MPC":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(os.path.join(SEC, "TabFrontier.tex"), "w").write("\n".join(lines))
    print("TabFrontier.tex written")


def coverage_table():
    agg = pd.read_csv(os.path.join(OUT, "twostage_agg.csv"))
    fr = load("frontier")
    lines = ["\\begin{table}[htbp]", "\\centering",
             "\\caption{Calibration behaviour across nominal confidence "
             "levels, replicated. Operational: the closed-loop adaptive "
             "scheme actually used by the controller (10 paired realisations; "
             "coverage of accepted promises). Locked: chronological "
             "commissioning/evaluation split with calibration frozen before "
             "evaluation, outcomes recorded for all requests including "
             "abstentions (30 realisations); single-stage calibrates "
             "magnitude only, two-stage predicts availability first and "
             "calibrates magnitude conditional on it. Withheld = fraction of "
             "evaluation requests for which the two-stage construction "
             "declines to certify a nonzero quantity.}",
             "\\label{tab:coverage}", "\\footnotesize",
             "\\setlength{\\tabcolsep}{3.5pt}",
             "\\begin{tabular}{ccccccc}", "\\toprule",
             " & & Operational & \\multicolumn{2}{c}{Locked single-stage} & "
             "\\multicolumn{2}{c}{Locked two-stage} \\\\",
             "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
             "$\\alpha$ & $1-\\alpha$ & coverage & coverage & mean cert.\\ (kW)"
             " & coverage & withheld (\\%) \\\\",
             "\\midrule"]
    opmap = {0.30: "Pa0.30", 0.20: "Pa0.20", 0.10: "Pa0.10",
             0.05: "Pa0.05", 0.02: "Pa0.02"}
    for _, r in agg.sort_values("alpha", ascending=False).iterrows():
        a = r.alpha
        op = fr[fr.controller == opmap[a]].coverage
        c2 = ("--" if np.isnan(r.cov_two)
              else f"{r.cov_two:.3f}\\,$\\pm$\\,{(0 if np.isnan(r.cov_two_sd) else r.cov_two_sd):.3f}")
        lines.append(
            f"{a:.2f} & {1-a:.2f} & {cell(op,3)} & "
            f"{r.cov_one:.3f}\\,$\\pm$\\,{r.cov_one_sd:.3f} & "
            f"{r.mean_cert_one:.0f} & {c2} & "
            f"{100*r.withheld_two:.1f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(os.path.join(SEC, "TabCoverage.tex"), "w").write("\n".join(lines))
    print("TabCoverage.tex written")


def ablation_table():
    df = load("ablation")
    labs = [("full", "Full method"), ("nocert", "$-$ de-rating layer"),
            ("norisk", "$-$ risk term"), ("noinfo", "$-$ information term"),
            ("nobins", "$-$ regime bins")]
    lines = ["\\begin{table}[htbp]", "\\centering",
             "\\caption{Case A ablation under the common external request: "
             "mean $\\pm$ 95\\% CI over 10 paired realisations.}",
             "\\label{tab:ablation}", "\\footnotesize",
             "\\setlength{\\tabcolsep}{3.5pt}",
             "\\begin{tabular}{lccccc}", "\\toprule",
             "Configuration & CO$_2$ & Service & False act. & Abstention & "
             "Net benefit \\\\",
             " & (t/yr) & rate (\\%) & (\\%) & (\\%) & (\\$/yr) \\\\",
             "\\midrule"]
    for tag, lbl in labs:
        g = df[df.controller == tag]
        lines.append(lbl + " & " + " & ".join([
            cell(g.CO2_t_per_yr, 1), cell(g.uncond_service_rate_pct, 1),
            cell(g.false_activation_pct, 1), cell(g.abstention_rate_pct, 1),
            cell(g.net_service_benefit, 0)]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(os.path.join(SEC, "TabAblation.tex"), "w").write("\n".join(lines))
    print("TabAblation.tex written")


def weights_table():
    p = os.path.join(OUT, "weights.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    lines = ["\\begin{table}[htbp]", "\\centering",
             "\\caption{Weight-sensitivity analysis: paired P$-$B5 "
             "differences (mean $\\pm$ 95\\% CI, 5 paired realisations per "
             "variant) when both controllers receive identically perturbed "
             "objective weights. The qualitative conclusion---fewer false "
             "activations and lower EV shortfall at the cost of a lower "
             "unconditional service rate---is stable in sign across all "
             "variants.}",
             "\\label{tab:weights}", "\\footnotesize",
             "\\setlength{\\tabcolsep}{3.5pt}",
             "\\begin{tabular}{lccc}", "\\toprule",
             "Variant & $\\Delta$ service rate (pp) & $\\Delta$ false act.\\ "
             "(pp) & $\\Delta$ EV shortfall (MWh/yr) \\\\",
             "\\midrule"]
    order = ["base", "carbon_x0.5", "carbon_x2", "comfort_x0.5", "comfort_x2",
             "service_x0.5", "service_x2", "ev_x0.5"]
    nice = {"base": "reported weights", "carbon_x0.5": "$w_C\\times 0.5$",
            "carbon_x2": "$w_C\\times 2$", "comfort_x0.5": "$w_T\\times 0.5$",
            "comfort_x2": "$w_T\\times 2$", "service_x0.5": "$w_G\\times 0.5$",
            "service_x2": "$w_G\\times 2$", "ev_x0.5": "$w_M\\times 0.5$"}
    for v in order:
        g = df[df.variant == v]
        if not len(g):
            continue
        dif = {}
        for m in ["uncond_service_rate_pct", "false_activation_pct",
                  "EV_shortfall_kWh"]:
            a = g[g.controller == "P"].set_index("seed")[m]
            b = g[g.controller == "B5"].set_index("seed")[m]
            dif[m] = (a - b).dropna().values
        lines.append(nice[v] + " & " + " & ".join([
            cell(dif["uncond_service_rate_pct"], 1),
            cell(dif["false_activation_pct"], 1),
            cell(dif["EV_shortfall_kWh"], 2, 1e-3)]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(os.path.join(SEC, "TabWeights.tex"), "w").write("\n".join(lines))
    print("TabWeights.tex written")


def runtime_table():
    p = os.path.join(OUT, "runtime.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    lines = ["\\begin{table}[htbp]", "\\centering",
             "\\caption{Measured runtime and memory. Median and 95th "
             "percentile of per-step solver wall-clock for the operational "
             "program, and separately for the auxiliary envelope program "
             "(proposed controller only); peak resident memory of the whole "
             "process. Hardware and software configuration in the text.}",
             "\\label{tab:runtime}", "\\footnotesize",
             "\\setlength{\\tabcolsep}{3.5pt}",
             "\\begin{tabular}{lccccccc}", "\\toprule",
             "Config & $H$ & clusters & op.\\ med (ms) & op.\\ p95 (ms) & "
             "aux.\\ med (ms) & aux.\\ p95 (ms) & RSS (MB) \\\\",
             "\\midrule"]
    for _, r in df.iterrows():
        nm = ("B5, 42 d" if r.which == "B5" and r.days == 42 else
              ("P, 42 d" if r.days == 42 else "P, 7 d"))
        aux = ("--" if r.env_n == 0 or np.isnan(r.env_med_ms)
               else f"{r.env_med_ms:.0f}")
        auxp = ("--" if r.env_n == 0 or np.isnan(r.env_p95_ms)
                else f"{r.env_p95_ms:.0f}")
        lines.append(f"{nm} & {int(r.H)} & {int(r.n_cl)} & "
                     f"{r.op_med_ms:.0f} & {r.op_p95_ms:.0f} & {aux} & "
                     f"{auxp} & {r.peak_rss_MB:.0f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(os.path.join(SEC, "TabRuntime.tex"), "w").write("\n".join(lines))
    print("TabRuntime.tex written")


if __name__ == "__main__":
    which = sys.argv[1:] or ["all"]
    if "all" in which or "main" in which:
        main_table("caseA", "TabMain.tex",
                   "Case A under the common external request and settlement "
                   "rule: mean $\\pm$ 95\\% CI over 30 paired realisations "
                   "(identical exogenous draws and request sequences for all "
                   "controllers within each realisation). Delivery is "
                   "measured against the common uncontrolled baseline; "
                   "settlement credits delivery up to the promise "
                   "(\\$0.09/kWh) and penalises shortfall against the promise "
                   "(\\$0.45/kWh, 5\\% tolerance); over-delivery is unpaid "
                   "and reported separately. Flexibility volumes are per "
                   "42-day campaign.",
                   "tab:main", ["B1", "B2", "B3", "B4", "B5", "B6", "P"],
                   ["B1", "B2", "B3", "B4", "B5", "B6", "\\textbf{P}"])
        frontier_table()
        coverage_table()
        ablation_table()
    if "all" in which or "caseC" in which:
        main_table("caseC", "TabCaseC.tex",
                   "Case C (repaired cold-climate, winter-peaking "
                   "configuration) under the common external request: mean "
                   "$\\pm$ 95\\% CI over 30 paired realisations. Format as "
                   "Table~\\ref{tab:main}.",
                   "tab:caseC", ["B1", "B2", "B3", "B4", "B5", "B6", "P"],
                   ["B1", "B2", "B3", "B4", "B5", "B6", "\\textbf{P}"])
    if "all" in which or "aux" in which:
        weights_table()
        runtime_table()
