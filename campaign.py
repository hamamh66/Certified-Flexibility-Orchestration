"""
Confirmatory campaign for NEXUS-D-26-00490.

Implements the reviewers' required protocol:
  * ONE externally specified grid-service request sequence per seed, sized
    against the uncontrolled (B1, no-response) baseline import, identical for
    every controller (same power, duration, timing, tolerance, price, and
    non-delivery penalty).
  * Delivery measured for every controller against the SAME common baseline.
  * Settlement under a common rule: revenue on delivered-up-to-promise,
    penalty on shortfall vs promise; over-delivery unpaid, reported separately.
  * Paired replication: all controllers face the identical realisation.
  * Outcomes recorded for ALL requests, including abstentions.
"""
import sys, os, time, json, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from cfo_lib import (BuildingCfg, FleetCfg, SimCfg, Scenario, MPCCore,
                     ConformalCertifier, Uncontrolled, TOURule, CarbonRule,
                     MPCController, CFOController, run, measure_delivery,
                     summarise)

DAYS = 42
REV_RATE = 0.09          # $/kWh on delivered-up-to-promise (common rule)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)


def cfo_factory(alpha=0.10, mode="adaptive", **kw):
    def f(s_, c_):
        cert = ConformalCertifier(alpha=alpha, mode=mode,
                                  min_n=s_.s.calib_min, lr=s_.s.adaptive_lr)
        return CFOController(s_, c_, cert, **kw)
    return f


def settlement(sc, deliv, log):
    """Common-rule service metrics over ALL externally issued requests."""
    o = {}
    scale = 365.0 / sc.s.days
    tol = sc.s.deliver_tol
    if not len(deliv):
        return o
    d = deliv
    dur = d["dur"].values
    req, prom, got = d["requested"].values, d["promised"].values, d["delivered"].values
    acc = prom > 1e-6
    credit = np.minimum(got, prom)
    short = np.maximum(prom - got, 0.0)
    over = np.where(acc, np.maximum(got - prom, 0.0), 0.0)
    o["n_requests"] = len(d)
    o["requested_MWh"] = float((req * dur).sum() / 1e3)
    o["accepted_MWh"] = float((prom * dur).sum() / 1e3)
    o["delivered_MWh"] = float((np.minimum(got, req) * dur).sum() / 1e3)
    o["acceptance_rate_pct"] = 100.0 * float(acc.mean())
    o["abstention_rate_pct"] = 100.0 * float(1 - acc.mean())
    o["cond_delivery_rate_pct"] = (100.0 * float(
        (got[acc] >= prom[acc] * (1 - tol) - 1e-6).mean()) if acc.any()
        else np.nan)
    o["uncond_service_rate_pct"] = 100.0 * float(
        (got >= req * (1 - tol) - 1e-6).mean())
    o["delivered_frac_of_requested"] = float(
        (np.minimum(got, req) * dur).sum() / max((req * dur).sum(), 1e-9))
    o["delivered_frac_of_accepted"] = (float(
        (credit[acc] * dur[acc]).sum() / max((prom[acc] * dur[acc]).sum(), 1e-9))
        if acc.any() else np.nan)
    o["shortfall_MWh"] = float((short * dur).sum() / 1e3)
    o["overdelivery_MWh"] = float((over * dur).sum() / 1e3)
    rev = (credit * dur).sum() * REV_RATE
    pen = (short * dur).sum() * sc.s.nd_penalty
    o["service_revenue_yr"] = float(rev * scale)
    o["service_penalty_yr"] = float(pen * scale)
    o["net_service_benefit"] = float((rev - pen) * scale)
    o["false_activation_pct"] = (100.0 * float(
        (got[acc] < prom[acc] * (1 - tol) - 1e-6).mean()) if acc.any()
        else np.nan)
    return o


def run_seed(seed, bcfg=None, controllers=None, days=DAYS, severity=1.0,
             keep_records=False):
    """One paired realisation: identical scenario + common request for all."""
    bcfg = bcfg or BuildingCfg()
    fcfg = FleetCfg()
    scfg = SimCfg(days=days, seed=seed, forecast_severity=severity)
    sc = Scenario(bcfg, fcfg, scfg)
    core = MPCCore(bcfg, fcfg, scfg)
    # common baseline: uncontrolled, no response
    _, cf = run(sc, lambda s_, c_: Uncontrolled(s_), core, respond=False)
    base = cf["imp"].values
    sc.resize_events(base)              # ONE external request sequence

    if controllers is None:
        controllers = [
            ("B1", lambda s_, c_: Uncontrolled(s_), False),
            ("B2", lambda s_, c_: TOURule(s_), False),
            ("B3", lambda s_, c_: CarbonRule(s_), False),
            ("B4", lambda s_, c_: MPCController(s_, c_, perfect=True), False),
            ("B5", lambda s_, c_: MPCController(s_, c_), False),
            ("B6", lambda s_, c_: MPCController(s_, c_, robust=0.20), False),
            ("P",  cfo_factory(alpha=0.10, mode="adaptive"), True),
        ]
    rows, records = [], {}
    for name, fac, fb in controllers:
        ctrl, log = run(sc, fac, core, feedback=fb, ref_import=base,
                        respond=True)
        d = measure_delivery(sc, log, ctrl, base)
        row = summarise(sc, log, d, name)
        row.update(settlement(sc, d, log))
        row["seed"] = seed
        rows.append(row)
        if keep_records:
            records[name] = dict(
                deliv=d,
                envelopes=getattr(ctrl, "envelopes", None),
                log_cols=log[["imp", "theta", "discomfort"]])
    return rows, records


def main_campaign(tag, seeds, bcfg=None, controllers=None, severity=1.0,
                  keep=()):
    all_rows, kept = [], {}
    t0 = time.time()
    for i, sd in enumerate(seeds):
        rows, rec = run_seed(sd, bcfg=bcfg, controllers=controllers,
                             severity=severity, keep_records=bool(keep))
        all_rows += rows
        if keep:
            kept[sd] = {k: v for k, v in rec.items() if k in keep}
        df = pd.DataFrame(all_rows)
        df.to_csv(os.path.join(OUT, f"{tag}.csv"), index=False)
        print(f"[{tag}] seed {sd} done ({i+1}/{len(seeds)}) "
              f"{time.time()-t0:.0f}s", flush=True)
    if keep:
        with open(os.path.join(OUT, f"{tag}_records.pkl"), "wb") as fh:
            pickle.dump(kept, fh)
    return pd.DataFrame(all_rows)


if __name__ == "__main__":
    what = sys.argv[1]
    seeds30 = [20260000 + i for i in range(30)]
    seeds10 = seeds30[:10]
    if what == "caseA":
        half = int(sys.argv[2])  # 0 or 1 -> split across 2 processes
        seeds = seeds30[half::2]
        main_campaign(f"caseA_{half}", seeds, keep=("P",))
    elif what == "frontier":
        ctr = []
        for r in [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]:
            ctr.append((f"B6r{r:.2f}",
                        (lambda rr: lambda s_, c_: MPCController(
                            s_, c_, robust=rr))(r), False))
        for a in [0.30, 0.20, 0.10, 0.05, 0.02]:
            ctr.append((f"Pa{a:.2f}", cfo_factory(alpha=a), True))
        half = int(sys.argv[2])
        main_campaign(f"frontier_{half}", seeds10[half::2], controllers=ctr)
    elif what == "ablation":
        ctr = [
            ("full",   cfo_factory(alpha=0.10, mode="adaptive"), True),
            ("nocert", cfo_factory(alpha=0.10, mode="none"), True),
            ("norisk", cfo_factory(alpha=0.10, use_risk=False), True),
            ("noinfo", cfo_factory(alpha=0.10, use_info=False), True),
            ("nobins", cfo_factory(alpha=0.10, mode="split"), True),
        ]
        half = int(sys.argv[2])
        main_campaign(f"ablation_{half}", seeds10[half::2], controllers=ctr)
    elif what == "stress":
        ctr = [("B5", lambda s_, c_: MPCController(s_, c_), False),
               ("P", cfo_factory(alpha=0.10), True)]
        sev = float(sys.argv[2])
        main_campaign(f"stress_{sev}", seeds10, controllers=ctr, severity=sev)
    elif what == "caseC":
        # repaired winter case: realistic envelope conductance (UA = 5 kW/K)
        # repaired winter case: cold-season representative days, realistic
        # envelope conductance (UA = 4 kW/K), heating-season comfort band with
        # adaptive upper limit (ASHRAE 55-style), no mechanical cooling.
        bc = BuildingCfg(name="CaseC", winter_peaking=True, mean_temp=-2.0,
                         lat_seasonal_amp=3.0, cop_heat=3.0, theta_lo=20.0,
                         theta_hi=27.0, theta_set=22.0, R=2.5e-4,
                         sol_gain_frac=0.10)
        half = int(sys.argv[2])
        main_campaign(f"caseC_{half}", seeds30[half::2], bcfg=bc, keep=("P",))
