"""Weight-sensitivity analysis (correction 1.4).

Both the proposed controller and the B5 comparator receive the SAME modified
weights in each variant (fair tuning), 5 paired seeds each. The reported
outcome is the sign and magnitude of the P-vs-B5 differences in unconditional
service rate, false activation and EV shortfall.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from cfo_lib import (BuildingCfg, FleetCfg, SimCfg, Scenario, MPCCore,
                     Uncontrolled, MPCController, run, measure_delivery,
                     summarise)
from campaign import cfo_factory, settlement, OUT

VARIANTS = {
    "base":        {},
    "carbon_x0.5": {"carbon": 0.5},
    "carbon_x2":   {"carbon": 2.0},
    "comfort_x0.5": {"comfort": 11.0},
    "comfort_x2":  {"comfort": 44.0},
    "service_x0.5": {"service": 20.0},
    "service_x2":  {"service": 80.0},
    "ev_x0.5":     {"ev": 1500.0},
}
rows = []
for vname, w in VARIANTS.items():
    for seed in [20260000 + i for i in range(5)]:
        bcfg, fcfg = BuildingCfg(), FleetCfg()
        scfg = SimCfg(days=42, seed=seed)
        sc = Scenario(bcfg, fcfg, scfg)
        core = MPCCore(bcfg, fcfg, scfg, weights=w)
        _, cf = run(sc, lambda s_, c_: Uncontrolled(s_), core, respond=False)
        base = cf["imp"].values
        sc.resize_events(base)
        for name, fac, fb in [
                ("B5", lambda s_, c_: MPCController(s_, c_), False),
                ("P", cfo_factory(alpha=0.10), True)]:
            ctrl, log = run(sc, fac, core, feedback=fb, ref_import=base,
                            respond=True)
            d = measure_delivery(sc, log, ctrl, base)
            r = summarise(sc, log, d, name)
            r.update(settlement(sc, d, log))
            r["seed"], r["variant"] = seed, vname
            rows.append(r)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "weights.csv"),
                                  index=False)
        print(vname, seed, "done", flush=True)
