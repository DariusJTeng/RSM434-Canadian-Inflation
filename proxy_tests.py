"""Test every major CPI component against the TRADABLE instruments actually in the hedge."""
import pandas as pd, numpy as np
import statsmodels.api as sm
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)

p = pd.read_csv("data/clean/monthly.csv", index_col=0, parse_dates=True)
IS_END = "2017-12-31"
dl = np.log(p.filter(like="cpi_")).diff(); dl.columns = dl.columns.str.replace("cpi_", "")
fx = np.log(p["usdcad"]).diff()

inst = pd.DataFrame({                                  # monthly log returns in CAD
    "BCOM Energy TR":        np.log(p["bbg_energy_tr_cad"]).diff(),      # Bloomberg, already CAD
    "BCOM Petroleum TR":     np.log(p["bbg_petroleum_tr_cad"]).diff(),
    "BCOM Agriculture TR":   np.log(p["bbg_agri_tr_cad"]).diff(),
    "BCOM Livestock TR":     np.log(p["bbg_live_tr_cad"]).diff(),
    "TSX Energy TR":         np.log(p["tsx_energy_tr"]).diff(),
    "BCOM Grains TR":        np.log(p["bbg_grains_tr_cad"]).diff(),
    "BCOM Natural Gas TR":   np.log(p["bbg_natgas_tr_cad"]).diff(),
    "TSX Real Estate TR":    np.log(p["tsx_re_tr"]).diff(),
    "GoC 5y yield (chg)":    p["goc_5y"].diff() / 100,   # drives fixed mortgage rates
    "WTI spot (CAD)":        np.log(p["wti_usd"]).diff() + fx,
})

def lag_reg(y, x, lags):
    X = pd.concat({f"L{l}": x.shift(l) for l in range(lags + 1)}, axis=1)
    m = pd.get_dummies(y.index.month, prefix="m", drop_first=True, dtype=float); m.index = y.index
    d = pd.concat([y.rename("y"), X, m], axis=1, sort=True).dropna()
    ins, oos = d.loc[:IS_END], d.loc[IS_END:].iloc[1:]
    fit = sm.OLS(ins["y"], sm.add_constant(ins.drop(columns="y"))).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    seas = sm.OLS(ins["y"], sm.add_constant(ins[m.columns])).fit()
    lagc = list(X.columns)
    pt = fit.t_test(" + ".join(lagc) + " = 0")
    F = fit.f_test(", ".join(f"{c} = 0" for c in lagc))
    pred = fit.predict(sm.add_constant(oos.drop(columns="y"), has_constant="add"))
    base = oos.index.month.map(ins.groupby(ins.index.month)["y"].mean())
    return {"in-sample": f"{ins.index[0]:%Y}-{ins.index[-1]:%Y}", "N": int(fit.nobs),
            "pass_through": float(pt.effect[0]), "t": float(pt.tvalue[0][0]),
            "F p-value": float(F.pvalue), "ΔR2 in-sample": fit.rsquared - seas.rsquared,
            "R2 out-of-sample": 1 - ((oos["y"] - pred) ** 2).sum() / ((oos["y"] - base) ** 2).sum()}

TESTS = [  # component, instrument, lags (shelter is slow -> 12 months)
    ("energy",   "BCOM Energy TR", 6), ("energy", "WTI spot (CAD)", 6), ("gasoline", "BCOM Energy TR", 6),
    ("gasoline", "BCOM Petroleum TR", 6), ("energy", "TSX Energy TR", 6),
    ("food",     "BCOM Agriculture TR", 6), ("food", "BCOM Livestock TR", 6),
    ("food", "BCOM Grains TR", 6), ("energy", "BCOM Natural Gas TR", 6), ("shelter", "BCOM Natural Gas TR", 6),
    ("shelter",  "TSX Real Estate TR", 12), ("shelter", "GoC 5y yield (chg)", 12),
]
T = pd.DataFrame({f"{c} CPI <- {x} (lags 0-{k})": lag_reg(dl[c], inst[x], k) for c, x, k in TESTS}).T
print(T.round(3).to_string())

# Shelter in LEVELS over 12 months: does a year of REIT returns / rate moves line up with shelter inflation?
yoy = pd.DataFrame({"shelter y/y": np.log(p["cpi_shelter"]).diff(12),
                    "REIT 12m return": np.log(p["tsx_re_tr"]).diff(12),
                    "GoC 5y 12m change": p["goc_5y"].diff(12)})
print("\nCorrelation of 12-month changes (2005-2026):\n", yoy.corr().round(2).iloc[0, 1:].to_string())
print("Same, lagged 12m (does last year's move predict this year's shelter inflation?):")
print(pd.concat([yoy["shelter y/y"], yoy[["REIT 12m return", "GoC 5y 12m change"]].shift(12)], axis=1)
      .corr().round(2).iloc[0, 1:].to_string())
