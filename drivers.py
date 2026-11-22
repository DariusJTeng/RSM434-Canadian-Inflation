"""Step 1 - CPI driver analysis on the clean monthly panel."""
import pandas as pd, numpy as np
import statsmodels.api as sm
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)

p = pd.read_csv("data/clean/monthly.csv", index_col=0, parse_dates=True)
IS_END = "2017-12-31"                                  # in-sample cut-off from the brief

# ---------------- A. Which components move headline CPI? ----------------
W = {"food": 16.93, "shelter": 28.30, "household": 12.80, "clothing": 4.50,      # 2025 basket
     "transport": 18.54, "health": 5.40, "recreation": 9.81, "alc_tob": 3.73}
dl = np.log(p.filter(like="cpi_")).diff().dropna()
dl.columns = dl.columns.str.replace("cpi_", "")
contrib = pd.DataFrame({k: w / 100 * dl[k] for k, w in W.items()})
total = contrib.sum(axis=1)
A = pd.DataFrame({
    "weight_%": pd.Series(W),
    "vol_ann_%": dl[list(W)].std() * np.sqrt(12) * 100,
    "share_monthly_var_%": contrib.apply(lambda c: c.cov(total)) / total.var() * 100,
    "share_cum_infl_%": contrib.sum() / total.sum() * 100,
    "share_cum_infl_2021-23_%": contrib.loc["2021":"2023"].sum() / total.loc["2021":"2023"].sum() * 100,
}).round(1).sort_values("share_monthly_var_%", ascending=False)
print("A. Driver decomposition (2004-2026)\n", A, "\n")
print("energy (gasoline+utilities) share of monthly var: %.1f%% at weight 6.7%%"
      % (100 * (0.0668 * dl["energy"]).cov(dl["headline"]) / dl["headline"].var()))

# ---------------- B. Does each driver have a tradable proxy? ----------------
fx = np.log(p["usdcad"]).diff()                            # CAD per USD, + = CAD weakens
px = pd.DataFrame({
    "oil_cad":  np.log(p["wti_usd"] * p["usdcad"]).diff(),
    "food_cad": np.log(p["imf_food_usd"] * p["usdcad"]).diff(),
    "usdcad":   fx,
    "us_cpi":   np.log(p["us_cpi"]).diff(),
})

def lag_reg(y, x, lags=6):
    X = pd.concat({f"L{l}": x.shift(l) for l in range(lags + 1)}, axis=1)
    m = pd.get_dummies(y.index.month, prefix="m", drop_first=True, dtype=float)
    m.index = y.index
    d = pd.concat([y.rename("y"), X, m], axis=1, sort=True).dropna()
    ins, oos = d.loc[:IS_END], d.loc[IS_END:].iloc[1:]
    fit = sm.OLS(ins["y"], sm.add_constant(ins.drop(columns="y"))).fit(
        cov_type="HAC", cov_kwds={"maxlags": 6})
    lagc = [c for c in X.columns]
    r2_seas = sm.OLS(ins["y"], sm.add_constant(ins[m.columns])).fit().rsquared
    # out-of-sample: how much of the month-to-month variation the IN-SAMPLE model explains
    pred = fit.predict(sm.add_constant(oos.drop(columns="y"), has_constant="add"))
    seas = ins.groupby(ins.index.month)["y"].mean()          # seasonal-only benchmark
    base = oos.index.month.map(seas)
    r2_oos = 1 - ((oos["y"] - pred) ** 2).sum() / ((oos["y"] - base) ** 2).sum()
    wald = fit.f_test(" + ".join(lagc) + " = 0")
    return {"R2_seasonal_only": r2_seas, "R2_in": fit.rsquared,
            "R2_added_by_proxy": fit.rsquared - r2_seas, "R2_oos_vs_seasonal": r2_oos,
            "pass_through": fit.params[lagc].sum(), "p_value": float(wald.pvalue)}

tests = [("gasoline", "oil_cad"), ("energy", "oil_cad"), ("transport", "oil_cad"),
         ("food", "food_cad"), ("food", "usdcad"), ("household", "usdcad"),
         ("clothing", "usdcad"), ("goods", "usdcad"),
         ("headline", "oil_cad"), ("headline", "us_cpi"), ("core_xfe", "us_cpi")]
B = pd.DataFrame({f"{c} <- {x}": lag_reg(dl[c], px[x]) for c, x in tests}).T.round(3)
print("\nB. Proxy regressions (lags 0-6, month dummies, in-sample 2004-2017)\n", B)

# ---------------- C. The rate side: what moves the Canadian real yield? ----------------
dy = p[["rrb_long", "goc_long", "us_tips10_real", "us_be10"]].diff()
d = dy.dropna()
for cols in (["goc_long"], ["goc_long", "us_tips10_real"]):
    for lab, part in (("in", d.loc[:IS_END]), ("oos", d.loc["2018":])):
        f = sm.OLS(part["rrb_long"], sm.add_constant(part[cols])).fit()
        print(f"C. d(rrb_long) ~ {cols}  [{lab}]  R2={f.rsquared:.2f}  "
              + "  ".join(f"{k}={v:.2f}" for k, v in f.params.items()))
