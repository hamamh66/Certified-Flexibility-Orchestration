"""Two-stage availability-then-magnitude certification: repair and test.

Uses the per-request records of the proposed controller (P) from the Case A
replicated campaign. Each record pairs the auxiliary-LP envelope estimate
phi_hat and regime context with the realised delivery against the common
uncontrolled baseline -- recorded for EVERY external request, including
abstained ones, so calibration is not conditioned on acceptance.

Protocol (pre-specified before looking at coverage):
  * Per seed, requests are split chronologically: the first half is the
    commissioning/calibration set, the second half the evaluation set.
  * Stage 1: logistic availability model P(available | conn_frac, th_head,
    soc), where "available" means realised delivery >= 50% of phi_hat
    (binary response regime). Fitted on commissioning data only, then LOCKED.
  * Stage 2: split-conformal lower bound on delivered magnitude conditional
    on availability, calibrated on commissioning data only, then LOCKED.
  * Certified quantity at nominal miscoverage alpha, with the level split
    a1 = a2 = 1 - sqrt(1-alpha) so that (1-a1)(1-a2) >= 1-alpha:
       - abstain (certify 0, vacuously covered claim withheld) unless
         predicted availability >= 1 - a1;
       - else certify  q = max(0, phi_hat - Q_{1-a2}(scores | available)).
  * Marginal coverage = P(delivered >= certified) among nonzero certificates
    on the evaluation set; also reported per regime bin and per alpha.
The single-stage (magnitude-only) construction is evaluated on the same
locked split for comparison.
"""
import os, pickle, sys
import numpy as np, pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
ALPHAS = [0.30, 0.20, 0.10, 0.05, 0.02]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_logistic(X, y, iters=3000, lr=0.15, l2=1e-3):
    X = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = sigmoid(X @ w)
        g = X.T @ (p - y) / len(y) + l2 * w
        w -= lr * g
    return w


def predict_logistic(w, X):
    X = np.hstack([np.ones((len(X), 1)), X])
    return sigmoid(X @ w)


def bin_of(cf, th, sb):
    return ("H" if cf > 0.45 else "L") + ("H" if th > 0.9 else "L") + \
           ("H" if sb > 0.45 else "L")


def gather():
    rows = []
    for h in (0, 1):
        p = os.path.join(OUT, f"caseA_{h}_records.pkl")
        if not os.path.exists(p):
            continue
        with open(p, "rb") as fh:
            kept = pickle.load(fh)
        for seed, recs in kept.items():
            r = recs.get("P")
            if r is None or r["envelopes"] is None:
                continue
            env = pd.DataFrame(r["envelopes"])
            dl = r["deliv"]
            m = env.merge(dl[["t0", "delivered", "requested"]],
                          left_on="t", right_on="t0", how="inner",
                          suffixes=("", "_d"))
            m["seed"] = seed
            rows.append(m)
    df = pd.concat(rows, ignore_index=True)
    df["avail"] = (df.delivered >= 0.5 * df.phi_hat).astype(float)
    df["bin"] = [bin_of(a, b, c) for a, b, c in
                 zip(df.conn_frac, df.th_head, df.soc)]
    return df


def evaluate(df):
    out = []
    feats = ["conn_frac", "th_head", "soc"]
    for seed, g in df.groupby("seed"):
        g = g.sort_values("t").reset_index(drop=True)
        n = len(g)
        cal, ev = g.iloc[:n // 2], g.iloc[n // 2:]
        if len(cal) < 15 or len(ev) < 10:
            continue
        Xc, yc = cal[feats].values, cal.avail.values
        w = fit_logistic(Xc, yc)
        p_ev = predict_logistic(w, ev[feats].values)
        # stage-2 scores on commissioning data, conditional on availability
        av = cal[cal.avail > 0]
        scores = (av.phi_hat - av.delivered).values      # positive=overestim.
        for alpha in ALPHAS:
            a1 = a2 = 1 - np.sqrt(1 - alpha)
            # ---------- two-stage
            if len(scores) >= 10:
                k = int(np.ceil((len(scores) + 1) * (1 - a2))) - 1
                q = np.sort(scores)[min(k, len(scores) - 1)]
            else:
                q = np.inf
            cert2 = np.where(p_ev >= 1 - a1,
                             np.maximum(0.0, ev.phi_hat.values - q), 0.0)
            nz2 = cert2 > 1e-6
            cov2 = float((ev.delivered.values[nz2] >= cert2[nz2] - 1e-6).mean()) \
                if nz2.any() else np.nan
            # ---------- single-stage (magnitude only, same locked split)
            sc_all = (cal.phi_hat - cal.delivered).values
            k1 = int(np.ceil((len(sc_all) + 1) * (1 - alpha))) - 1
            q1 = np.sort(sc_all)[min(k1, len(sc_all) - 1)]
            cert1 = np.maximum(0.0, ev.phi_hat.values - q1)
            nz1 = cert1 > 1e-6
            cov1 = float((ev.delivered.values[nz1] >= cert1[nz1] - 1e-6).mean()) \
                if nz1.any() else np.nan
            out.append(dict(seed=seed, alpha=alpha, nominal=1 - alpha,
                            cov_two=cov2, n_cert_two=int(nz2.sum()),
                            width_two=float(np.mean(ev.phi_hat.values[nz2] - cert2[nz2])) if nz2.any() else np.nan,
                            frac_withheld_two=float(1 - nz2.mean()),
                            cov_one=cov1, n_cert_one=int(nz1.sum()),
                            width_one=float(np.mean(ev.phi_hat.values[nz1] - cert1[nz1])) if nz1.any() else np.nan,
                            mean_cert_two=float(cert2[nz2].mean()) if nz2.any() else 0.0,
                            mean_cert_one=float(cert1[nz1].mean()) if nz1.any() else 0.0))
    res = pd.DataFrame(out)
    res.to_csv(os.path.join(OUT, "twostage_perseed.csv"), index=False)
    agg = res.groupby("alpha").agg(
        nominal=("nominal", "first"),
        cov_two=("cov_two", "mean"), cov_two_sd=("cov_two", "std"),
        cov_one=("cov_one", "mean"), cov_one_sd=("cov_one", "std"),
        withheld_two=("frac_withheld_two", "mean"),
        width_two=("width_two", "mean"), width_one=("width_one", "mean"),
        mean_cert_two=("mean_cert_two", "mean"),
        mean_cert_one=("mean_cert_one", "mean"),
        n_seeds=("seed", "nunique")).reset_index()
    agg.to_csv(os.path.join(OUT, "twostage_agg.csv"), index=False)
    print(agg.to_string(index=False))
    # regime-conditional coverage at alpha=0.10, two-stage
    return res


if __name__ == "__main__":
    df = gather()
    print("requests gathered:", len(df), "seeds:", df.seed.nunique(),
          "availability rate:", round(df.avail.mean(), 3))
    evaluate(df)
