"""Step 2-3: value the liability, build the hedges, test them out of sample."""
import pandas as pd, numpy as np
from scipy.optimize import minimize
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)

p   = pd.read_csv("data/clean/monthly.csv", index_col=0, parse_dates=True)
rrb = pd.read_csv("data/clean/rrb_prices.csv", parse_dates=["date"])

FACE      = 100.0                    # $100M x CPI_T / CPI_0
IS_START  = "2004-04-30"             # full Bloomberg history (CPI lags need 3 months)
IS_END    = "2017-12-31"
OOS_START = pd.Timestamp("2017-12-31")   # liability starts here, pays 10y later
END       = p.index.max()

# ---------------------------------------------------------------- helpers
def bond_price(y, years, cpn):
    """Price per 100 of a semi-annual bond, yield y and coupon cpn in % (fractional periods ok)."""
    n = years * 2                                   # periods to maturity
    first = n - np.floor(n) if n % 1 else 1.0       # periods to the next coupon
    k = np.arange(first, n + 1e-9, 1.0)
    cf = np.full(len(k), cpn / 2); cf[-1] += 100
    return (cf / (1 + y / 200) ** k).sum()

def cm_bond_return(yld, maturity):
    """Monthly total return of a constant-maturity par bond rebuilt from a yield series (%)."""
    out = pd.Series(np.nan, index=yld.index)
    for t0, t1 in zip(yld.index[:-1], yld.index[1:]):
        y0, y1 = yld[t0], yld[t1]
        if np.isnan(y0) or np.isnan(y1): continue
        p1 = bond_price(y1, maturity - 1 / 12, y0)       # bought at par 1 month ago
        out[t1] = p1 / 100 - 1                           # dirty price already holds the accrual
    return out

def zero_price(y, years):              # liability discounting: semi-annual, like RRB yields
    return (1 + y / 200) ** (-2 * years)

# ---------------------------------------------------------------- 1. CPI known at each month-end
cpi_known = p["cpi_headline"].shift(1)          # month m's CPI is released ~3rd week of m+1
r_real    = p["rrb_long"]                       # long RRB real yield (BoC)

# ---------------------------------------------------------------- 2. the liability
def liability(start, years=10.0):
    idx = p.loc[start:].index
    tau = pd.Series([years - (t - idx[0]).days / 365.25 for t in idx], index=idx)
    L = FACE * cpi_known[idx] / cpi_known[start] * zero_price(r_real[idx], tau)
    dur = tau / (1 + r_real[idx] / 200)          # modified duration of a zero
    return pd.DataFrame({"L": L, "tau": tau, "dur": dur})

def cm_liability_return():
    """Return of a CONSTANT 10-year liability (for in-sample estimation)."""
    infl = cpi_known / cpi_known.shift(1)
    px   = zero_price(r_real, 10 - 1 / 12) / zero_price(r_real.shift(1), 10)
    return infl * px - 1

LIAB = liability(OOS_START)

# ---------------------------------------------------------------- 3. asset returns (CAD, monthly)
cash_ca = p["ca_tbill_bbg"].fillna(p["ca_tbill_3m"])           # Canada 3M T-bill (Bloomberg GTCAD3MO)
cash_us = p["us_tbill_3m"]
carry   = (cash_ca - cash_us) / 1200                  # FX-hedge carry for USD assets
fx_ret  = p["usdcad"].pct_change()                    # + = CAD weakens

R = pd.DataFrame({
    "cash":       cash_ca.shift(1) / 1200,
    "nom_10y":    cm_bond_return(p["goc_10y"], 10),          # nominal GoC (can be shorted)
    "nom_long":   cm_bond_return(p["goc_long"], 30),
    "tips_hdg":   p["us_tips_tr_usd"].pct_change() + carry.shift(1),       # Bloomberg TIPS TR, CAD-hedged
    "energy":     p["bbg_petroleum_tr_cad"].pct_change(),    # BCOM PETROLEUM (tracks gasoline CPI best), CAD
    "agri":       p["bbg_agri_tr_cad"].pct_change(),
    "livestock":  p["bbg_live_tr_cad"].pct_change(),
    "tsx_energy": p["tsx_energy_tr"].pct_change(),           # Canadian oil & gas equities
    "tsx_re":     p["tsx_re_tr"].pct_change(),
})
RATE_ASSETS = ["nom_10y", "nom_long", "tips_hdg"]      # scaled with liability duration OOS
SHORTABLE   = ["nom_10y", "nom_long"]

# ---------------------------------------------------------------- 4a. RRB benchmark (duration-matched)
rrb_tr  = rrb.pivot(index="date", columns="issue", values="total_return")
rrb_dur = rrb.pivot(index="date", columns="issue", values="mod_dur")

def rrb_weights(t, target_dur):
    d = rrb_dur.loc[t].dropna()
    d = d[d > 0.25]                                   # skip bonds about to mature
    lo, hi = d[d <= target_dur], d[d >= target_dur]
    if lo.empty: return pd.Series({hi.idxmin(): 1.0})
    if hi.empty: return pd.Series({lo.idxmax(): 1.0})
    a, b = lo.idxmax(), hi.idxmin()                   # two bonds that bracket the target
    if a == b: return pd.Series({a: 1.0})
    w_b = (target_dur - d[a]) / (d[b] - d[a])
    return pd.Series({a: 1 - w_b, b: w_b})

def run_rrb(dur_series):
    ret = pd.Series(np.nan, index=dur_series.index)
    for t0, t1 in zip(dur_series.index[:-1], dur_series.index[1:]):
        w = rrb_weights(t0, dur_series[t0])
        ret[t1] = (w * rrb_tr.loc[t1, w.index]).sum()
    return ret


# ---------------------------------------------------------------- 4c. RRB variants: the liability's rate risk sits at the LONG end
def run_rrb_long(dur_series):
    """Liability is discounted at the LONG RRB yield, so its risk sits at the long end:
    hold the longest RRB sized to the liability's duration, rest in CAD T-bills."""
    ret = pd.Series(np.nan, index=dur_series.index)
    for t0, t1 in zip(dur_series.index[:-1], dur_series.index[1:]):
        d = rrb_dur.loc[t0].dropna(); longest = d.idxmax()
        w = min(dur_series[t0] / d[longest], 1.0)
        ret[t1] = w * rrb_tr.loc[t1, longest] + (1 - w) * R.loc[t1, "cash"]
    return ret

def run_rrb_barbell(dur_series):
    """Longest RRB for the long-yield exposure + shortest RRB (instead of cash) so that
    100% of the hedge stays CPI-linked. Weights solve: value = L, duration = liability's."""
    ret = pd.Series(np.nan, index=dur_series.index)
    for t0, t1 in zip(dur_series.index[:-1], dur_series.index[1:]):
        d = rrb_dur.loc[t0].dropna(); d = d[d > 0.25]
        lo, hi = d.idxmin(), d.idxmax()
        w_hi = np.clip((dur_series[t0] - d[lo]) / (d[hi] - d[lo]), 0, 1)
        ret[t1] = w_hi * rrb_tr.loc[t1, hi] + (1 - w_hi) * rrb_tr.loc[t1, lo]
    return ret

# ---------------------------------------------------------------- 4b. proxy hedge (in-sample weights)
rL_cm = cm_liability_return()
ins = pd.concat([rL_cm.rename("L"), R], axis=1).loc[IS_START:IS_END].dropna()
X, y = ins[R.columns].values, ins["L"].values

def te(w): return np.std(y - X @ w)
cons   = [{"type": "eq", "fun": lambda w: w.sum() - 1}]               # fully funded
bounds = [(None, None) if c in SHORTABLE else (0, None) for c in R.columns]
w0 = np.zeros(len(R.columns)); w0[0] = 1
w_proxy = pd.Series(minimize(te, w0, bounds=bounds, constraints=cons, method="SLSQP").x,
                    index=R.columns)

# rates-only proxy (no commodities / equities) - shows what the "inflation" assets add
rates_only = ["cash"] + RATE_ASSETS
X2 = ins[rates_only].values
w2 = minimize(lambda w: np.std(y - X2 @ w), np.r_[1, 0, 0, 0],
              bounds=[(0, None), (None, None), (None, None), (0, None)],
              constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], method="SLSQP").x
w_rates = pd.Series(0.0, index=R.columns); w_rates[rates_only] = w2

D_IS = 10 / (1 + ins.index.map(lambda t: r_real[t]).values.mean() / 200)   # in-sample liability dur

def run_proxy(w_is, dur_series):
    """Each month: scale rate positions by liability duration / in-sample duration; cash = rest."""
    ret = pd.Series(np.nan, index=dur_series.index)
    for t0, t1 in zip(dur_series.index[:-1], dur_series.index[1:]):
        w = w_is.copy()
        w[RATE_ASSETS] *= dur_series[t0] / D_IS
        w["cash"] = 1 - w.drop("cash").sum()
        if w["cash"] < 0:                               # stay long-only in cash: trim nominal shorts
            w["cash"] = 0; w = w / w.sum()
        ret[t1] = (w * R.loc[t1]).sum()
    return ret

# ---------------------------------------------------------------- 5. out-of-sample test
NOM10_DUR = 8.5                                       # approx. mod. duration of a 10y par GoC bond
rL = LIAB["L"].pct_change()
H = pd.DataFrame({
    "Cash only":                R["cash"],
    "Nominal GoC (dur-matched)": run_proxy(pd.Series({"nom_10y": D_IS / NOM10_DUR}
                                                      ).reindex(R.columns, fill_value=0), LIAB["dur"]),
    "RRB dur-matched (brief)":  run_rrb(LIAB["dur"]),
    "RRB longest + cash":       run_rrb_long(LIAB["dur"]),
    "RRB barbell long+short":   run_rrb_barbell(LIAB["dur"]),
    "Proxy: rates only":        run_proxy(w_rates, LIAB["dur"]),
    "Proxy: rates+commod":      run_proxy(w_proxy, LIAB["dur"]),
}).loc[LIAB.index].iloc[1:]

def report(hret):
    L0 = LIAB["L"].iloc[0]
    hv = L0 * (1 + hret).cumprod()                     # hedge value, starts = liability value
    dL, dH = LIAB["L"].diff().iloc[1:], hv.diff().fillna(hv.iloc[0] - L0)
    gap = dH - dL                                      # monthly $ tracking error
    out = {"TE $M/yr": gap.std() * np.sqrt(12),
           "TE % of L": (hret - rL.iloc[1:]).std() * np.sqrt(12) * 100,
           "Shortfall end $M": hv.iloc[-1] - LIAB["L"].iloc[-1]}
    for a, b in [("2018", "2020"), ("2021", "2022"), ("2023", "2026")]:
        out[f"TE % {a}-{b}"] = (hret - rL.iloc[1:]).loc[a:b].std() * np.sqrt(12) * 100
    out["worst month $M"] = gap.min()
    return pd.Series(out)

if __name__ == "__main__":
    print("Liability:", LIAB.iloc[[0, 48, -1]].round(2).to_string(), "\n")
    print("Proxy weights (in-sample, fraction of liability):\n",
          pd.DataFrame({"rates+commod": w_proxy, "rates only": w_rates}).round(3).to_string(), "\n")
    print(f"In-sample TE (annual, % of L): rates+commod {te(w_proxy.values)*np.sqrt(12)*100:.2f}  "
          f"rates only {np.std(y - X2 @ w2)*np.sqrt(12)*100:.2f}\n")
    print("OUT OF SAMPLE  (liability starts Dec-2017, $100M face)\n",
          H.apply(report).T.round(2).to_string())
    pd.concat([LIAB, H], axis=1).to_csv("out/hedge_results.csv")
