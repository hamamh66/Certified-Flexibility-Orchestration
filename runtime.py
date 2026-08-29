"""Runtime, scalability and memory measurement (correction 1.5).

Instruments per-step wall-clock of the operational MPC solve and the
auxiliary envelope solve separately, over a full 42-day Case A run, for the
B5 baseline and the proposed controller; then sweeps horizon length and
EV-cluster count. Reports median and 95th percentile, plus peak RSS.
Run on an otherwise idle machine (after the campaign workers finish).
"""
import os, sys, time, json, resource, platform, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import cvxpy
from cfo_lib import (BuildingCfg, FleetCfg, SimCfg, Scenario, MPCCore,
                     ConformalCertifier, Uncontrolled, MPCController,
                     CFOController, run)
from campaign import cfo_factory

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def instrument(core):
    op_times, env_times = [], []
    orig_solve, orig_env = core.solve, core.envelope

    def solve(*a, **k):
        t0 = time.perf_counter()
        r = orig_solve(*a, **k)
        op_times.append(time.perf_counter() - t0)
        return r

    def envelope(*a, **k):
        t0 = time.perf_counter()
        r = orig_env(*a, **k)
        env_times.append(time.perf_counter() - t0)
        return r
    core.solve, core.envelope = solve, envelope
    return op_times, env_times


def one(days, seed, horizon, cluster_sizes, which="P"):
    bcfg = BuildingCfg()
    fcfg = FleetCfg(n_clusters=len(cluster_sizes),
                    cluster_sizes=tuple(cluster_sizes),
                    profiles=tuple([(8.0, 17.5, 0.55, 0.80, 0.85)] *
                                   len(cluster_sizes)))
    scfg = SimCfg(days=days, seed=seed, horizon=horizon)
    sc = Scenario(bcfg, fcfg, scfg)
    core = MPCCore(bcfg, fcfg, scfg)
    _, cf = run(sc, lambda s_, c_: Uncontrolled(s_), core, respond=False)
    base = cf["imp"].values
    sc.resize_events(base)
    op, env = instrument(core)
    t0 = time.time()
    if which == "P":
        run(sc, cfo_factory(alpha=0.10), core, feedback=True,
            ref_import=base, respond=True)
    else:
        run(sc, lambda s_, c_: MPCController(s_, c_), core,
            ref_import=base, respond=True)
    wall = time.time() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # MB
    q = lambda a, p: float(np.percentile(a, p)) if a else np.nan
    return dict(which=which, days=days, H=horizon, n_cl=len(cluster_sizes),
                n_steps=len(op),
                op_med_ms=1e3 * q(op, 50), op_p95_ms=1e3 * q(op, 95),
                env_n=len(env), env_med_ms=1e3 * q(env, 50),
                env_p95_ms=1e3 * q(env, 95),
                wall_s=wall, per_step_ms=1e3 * wall / max(len(op), 1),
                peak_rss_MB=rss)


if __name__ == "__main__":
    rows = []
    # principal configuration, both controllers
    for w in ("B5", "P"):
        rows.append(one(42, 20260000, 24, [10, 8, 5], which=w))
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "runtime.csv"), index=False)
        print(rows[-1], flush=True)
    # horizon sweep (proposed controller, 7 days for tractability)
    for H in (12, 24, 48, 96):
        rows.append(one(7, 20260001, H, [10, 8, 5], which="P"))
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "runtime.csv"), index=False)
        print(rows[-1], flush=True)
    # cluster sweep
    for ncl in (1, 3, 6, 12):
        sizes = [23 // ncl + (1 if i < 23 % ncl else 0) for i in range(ncl)]
        rows.append(one(7, 20260002, 24, sizes, which="P"))
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "runtime.csv"), index=False)
        print(rows[-1], flush=True)
    env = dict(python=platform.python_version(), cvxpy=cvxpy.__version__,
               numpy=np.__version__,
               solver="CLARABEL (SCS fallback)",
               cpu=subprocess.getoutput("grep -m1 'model name' /proc/cpuinfo"),
               n_cores=os.cpu_count(),
               os_=platform.platform(),
               threads=os.environ.get("OMP_NUM_THREADS", "default"))
    json.dump(env, open(os.path.join(OUT, "runtime_env.json"), "w"), indent=1)
    print(env)
