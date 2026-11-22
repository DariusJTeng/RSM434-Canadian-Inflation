"""Rolling start-date test (walk-forward, no look-ahead).
For each start month s: estimate proxy weights on 2004..s only, start a fresh 10-year liability at s,
hold every hedge for HORIZON years, and record tracking error + shortfall."""
import pandas as pd, numpy as np
from scipy.optimize import minimize
import hedge as h

HORIZON = 5                          # years each test runs
MIN_IS  = 5                          # minimum years of estimation data
FIRST   = pd.Timestamp(h.IS_START) + pd.DateOffset(years=MIN_IS)
LAST    = h.END - pd.DateOffset(years=HORIZON)
starts  = [t for t in h.p.index if FIRST <= t <= LAST]

R, rL_cm, RATE, SHORT = h.R, h.rL_cm, h.RATE_ASSETS, h.SHORTABLE

def estimate(end, assets):
    ins = pd.concat([rL_cm.rename("L"), R[assets]], axis=1).loc[h.IS_START:end].dropna()
    X, y = ins[assets].values, ins["L"].values
    bnds = [(None, None) if a in SHORT else (0, None) for a in assets]
    w0 = np.zeros(len(assets)); w0[assets.index("cash")] = 1
    w = minimize(lambda w: np.std(y - X @ w), w0, bounds=bnds, method="SLSQP",
                 constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}]).x
    d_is = 10 / (1 + h.r_real.loc[ins.index].mean() / 200)
    return pd.Series(w, index=assets).reindex(R.columns, fill_value=0.0), d_is

def run(w_is, d_is, dur):
    ret = pd.Series(np.nan, index=dur.index)
    for t0, t1 in zip(dur.index[:-1], dur.index[1:]):
        w = w_is.copy(); w[RATE] *= dur[t0] / d_is; w["cash"] = 1 - w.drop("cash").sum()
        if w["cash"] < 0: w["cash"] = 0; w = w / w.sum()
        ret[t1] = (w * R.loc[t1]).sum()
    return ret

def score(hret, L):
    hv = L["L"].iloc[0] * (1 + hret.iloc[1:]).cumprod()
    rl = L["L"].pct_change().iloc[1:]
    return {"TE %": (hret.iloc[1:] - rl).std() * np.sqrt(12) * 100,
            "shortfall % of L": (hv.iloc[-1] / L["L"].iloc[-1] - 1) * 100}

ALL   = list(R.columns)
RATES = ["cash"] + RATE
rows, weights = [], []
for s in starts:
    end = s + pd.DateOffset(years=HORIZON)
    L = h.liability(s).loc[:end]
    w_all, d_all = estimate(s, ALL)
    w_rat, d_rat = estimate(s, RATES)
    weights.append(w_all.rename(s))
    strat = {"Cash only": R["cash"].loc[L.index],
             "RRB dur-matched": h.run_rrb(L["dur"]),
             "Proxy: rates only": run(w_rat, d_rat, L["dur"]),
             "Proxy: rates+commod": run(w_all, d_all, L["dur"])}
    for name, r in strat.items():
        rows.append({"start": s, "hedge": name, **score(r, L)})

res = pd.DataFrame(rows)
res.to_csv("out/rolling_results.csv", index=False)
pd.DataFrame(weights).to_csv("out/rolling_weights.csv")

if __name__ == "__main__":
    print(f"{len(starts)} start dates, {starts[0]:%Y-%m} to {starts[-1]:%Y-%m}, {HORIZON}-year horizon\n")
    piv = res.pivot_table(index="start", columns="hedge", values="shortfall % of L")
    print("Shortfall (% of liability at end of window):")
    print(piv.describe().loc[["mean", "std", "min", "50%", "max"]].round(2).to_string(), "\n")
    print("Tracking error (% p.a.):")
    print(res.pivot_table(index="start", columns="hedge", values="TE %").describe()
          .loc[["mean", "min", "max"]].round(2).to_string(), "\n")
    print("Selected starts (shortfall %):")
    print(piv.loc[[t for t in piv.index if t.month == 12 or t == piv.index[0]]].round(2).to_string())
    gap = piv["Proxy: rates+commod"] - piv["Proxy: rates only"]
    print(f"\nCommodity sleeve helped in {(gap > 0).mean()*100:.0f}% of windows; "
          f"best {gap.max():+.2f}% ({gap.idxmax():%Y-%m}), worst {gap.min():+.2f}% ({gap.idxmin():%Y-%m})")
