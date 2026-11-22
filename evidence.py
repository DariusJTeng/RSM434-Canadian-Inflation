"""Evidence pack: regression tables + correlation tables + charts for the report/deck."""
import pandas as pd, numpy as np
import statsmodels.api as sm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import hedge                                   # re-uses liability + asset returns (R, rL_cm)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
OUT = Path("out"); OUT.mkdir(exist_ok=True)
p = hedge.p
IS_END = "2017-12-31"
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1"

# ============================================================ data
dl = np.log(p.filter(like="cpi_")).diff()
dl.columns = dl.columns.str.replace("cpi_", "")
px = pd.DataFrame({
    "Oil (WTI, CAD)":          np.log(p["wti_usd"] * p["usdcad"]).diff(),
    "Food commodities (CAD)":  np.log(p["imf_food_usd"] * p["usdcad"]).diff(),
    "USD/CAD":                 np.log(p["usdcad"]).diff(),
    "US CPI":                  np.log(p["us_cpi"]).diff(),
})
PAIRS = [("gasoline", "Oil (WTI, CAD)"), ("energy", "Oil (WTI, CAD)"),
         ("headline", "Oil (WTI, CAD)"),
         ("food", "Food commodities (CAD)"), ("food", "USD/CAD"),
         ("household", "USD/CAD"), ("goods", "USD/CAD"),
         ("headline", "US CPI"), ("core_xfe", "US CPI")]

# ============================================================ TABLE 1: distributed-lag regressions
def lag_reg(y, x, lags=6):
    X = pd.concat({f"L{l}": x.shift(l) for l in range(lags + 1)}, axis=1)
    m = pd.get_dummies(y.index.month, prefix="m", drop_first=True, dtype=float); m.index = y.index
    d = pd.concat([y.rename("y"), X, m], axis=1, sort=True).dropna()
    ins, oos = d.loc[:IS_END], d.loc[IS_END:].iloc[1:]
    fit = sm.OLS(ins["y"], sm.add_constant(ins.drop(columns="y"))).fit(
        cov_type="HAC", cov_kwds={"maxlags": 6})
    seas = sm.OLS(ins["y"], sm.add_constant(ins[m.columns])).fit()
    lagc = list(X.columns)
    pt = fit.t_test(" + ".join(lagc) + " = 0")               # pass-through = sum of lag coefs
    F = fit.f_test(", ".join(f"{c} = 0" for c in lagc))     # all lags jointly zero
    pred = fit.predict(sm.add_constant(oos.drop(columns="y"), has_constant="add"))
    base = oos.index.month.map(ins.groupby(ins.index.month)["y"].mean())
    return {"N": int(fit.nobs),
            "pass_through": float(pt.effect[0]), "SE (HAC)": float(pt.sd[0][0]),
            "t": float(pt.tvalue[0][0]), "F p-value (all lags=0)": float(F.pvalue),
            "R2 seasonal only": seas.rsquared, "R2 with proxy": fit.rsquared,
            "ΔR2 (in-sample)": fit.rsquared - seas.rsquared,
            "R2 out-of-sample": 1 - ((oos["y"] - pred) ** 2).sum() / ((oos["y"] - base) ** 2).sum()}

T1 = pd.DataFrame({f"{c} CPI  <-  {x}": lag_reg(dl[c], px[x]) for c, x in PAIRS}).T

# ============================================================ TABLE 2: each hedge asset vs the liability
L = hedge.rL_cm.rename("liability")
def vs_liability(asset):
    d = pd.concat([L, hedge.R[asset]], axis=1).dropna()
    out = {}
    for lab, part in (("2012-17", d.loc[hedge.IS_START:IS_END]), ("2018-26", d.loc["2018":])):
        f = sm.OLS(part["liability"], sm.add_constant(part[asset])).fit(
            cov_type="HAC", cov_kwds={"maxlags": 3})
        out[f"corr {lab}"] = part.corr().iloc[0, 1]
        out[f"beta {lab}"] = f.params[asset]
        out[f"t {lab}"] = f.tvalues[asset]
    out["ann. vol %"] = d[asset].std() * np.sqrt(12) * 100
    return out
T2 = pd.DataFrame({a: vs_liability(a) for a in hedge.R.columns if a != "cash"}).T
T2.loc["(liability itself)", "ann. vol %"] = L.std() * np.sqrt(12) * 100

# ---- and vs the INFLATION part of the liability only (strip out the rate effect)
infl = hedge.cpi_known.pct_change().rename("cpi_known_infl")
T3 = pd.DataFrame({a: {
        "corr same month": pd.concat([infl, hedge.R[a]], axis=1).loc["2012":].corr().iloc[0, 1],
        "max |corr| lags 0-6": max(abs(pd.concat([infl, hedge.R[a].shift(k)], axis=1)
                                   .loc["2012":].corr().iloc[0, 1]) for k in range(7))}
      for a in ["energy", "agri", "livestock", "tsx_re", "tips_hdg"]}).T

with pd.ExcelWriter(OUT / "evidence_tables.xlsx") as xw:
    T1.round(3).to_excel(xw, sheet_name="T1 driver regressions")
    T2.round(3).to_excel(xw, sheet_name="T2 assets vs liability")
    T3.round(3).to_excel(xw, sheet_name="T3 assets vs CPI infl")

# ============================================================ charts
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})

# Fig 1: explanatory power, in-sample vs out-of-sample
f1 = T1[["ΔR2 (in-sample)", "R2 out-of-sample"]].iloc[::-1]
fig, ax = plt.subplots(figsize=(9, 5.2))
yy = np.arange(len(f1)); h = 0.36
ax.barh(yy + h / 2, f1.iloc[:, 0], h, color=BLUE, label="Extra R² from proxy, in-sample 2004–17")
ax.barh(yy - h / 2, f1.iloc[:, 1].clip(lower=-0.05), h, color=ORANGE, label="R² out-of-sample 2018–26")
ax.set_yticks(yy, f1.index); ax.axvline(0, color=MUTED, lw=1)
ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
ax.set_xlabel("Share of monthly variation explained (beyond seasonality)")
ax.set_title("Only oil (and US CPI, via energy) explains Canadian CPI",
             loc="left", color=INK, fontsize=12)
ax.legend(frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "fig1_explanatory_power.png", dpi=200); plt.close(fig)

# Fig 2: the strong vs the weak link, side by side
def deseason(s):
    return s - s.groupby(s.index.month).transform("mean")
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=False)
for ax, (comp, prox, lags, title) in zip(axes, [
        ("gasoline", "Oil (WTI, CAD)", 1, "Gasoline CPI vs oil (same + previous month)"),
        ("food", "Food commodities (CAD)", 6, "Food CPI vs food commodities (last 6 months)")]):
    x = sum(px[prox].shift(k) for k in range(lags + 1)) * 100
    y = deseason(dl[comp]) * 100
    d = pd.concat([x, y], axis=1).dropna()
    r = d.corr().iloc[0, 1]
    ax.scatter(d.iloc[:, 0], d.iloc[:, 1], s=14, color=BLUE, alpha=0.55, edgecolor="none")
    b = np.polyfit(d.iloc[:, 0], d.iloc[:, 1], 1); xs = np.linspace(d.iloc[:, 0].min(), d.iloc[:, 0].max(), 50)
    ax.plot(xs, np.polyval(b, xs), color=ORANGE, lw=2)
    ax.set_title(f"{title}\ncorrelation = {r:.2f}", loc="left", fontsize=10.5, color=INK)
    ax.set_xlabel(f"{prox} change, %"); ax.set_ylabel(f"{comp.title()} CPI m/m, % (de-seasonalised)")
    ax.grid(True, color=GRID); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(OUT / "fig2_strong_vs_weak.png", dpi=200); plt.close(fig)

# Fig 3: correlation of each hedge asset with the liability
f3 = T2.drop(index="(liability itself)")[["corr 2012-17", "corr 2018-26"]]
names = {"nom_10y": "Nominal GoC 10y", "nom_long": "Nominal GoC long", "tips_hdg": "US TIPS (CAD-hdg)",
         "energy": "BCOM Energy", "agri": "BCOM Agriculture", "livestock": "BCOM Livestock",
         "tsx_re": "TSX Real Estate"}
f3 = f3.rename(index=names).iloc[::-1]
fig, ax = plt.subplots(figsize=(8.5, 4.6))
yy = np.arange(len(f3))
ax.barh(yy + h / 2, f3.iloc[:, 0], h, color=BLUE, label="2012–17 (in-sample)")
ax.barh(yy - h / 2, f3.iloc[:, 1], h, color=ORANGE, label="2018–26 (out-of-sample)")
ax.set_yticks(yy, f3.index); ax.axvline(0, color=MUTED, lw=1); ax.set_xlim(-1, 1)
ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
ax.set_xlabel("Correlation of monthly returns with the liability")
ax.set_title("Bonds and REITs move with the liability; commodities don't", loc="left",
             fontsize=12, color=INK)
ax.legend(frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig(OUT / "fig3_corr_with_liability.png", dpi=200); plt.close(fig)

if __name__ == "__main__":
    print("TABLE 1 - distributed-lag regressions (lags 0-6, month dummies, HAC SEs)\n", T1.round(3), "\n")
    print("TABLE 2 - each asset vs the liability's monthly return\n", T2.round(2), "\n")
    print("TABLE 3 - each asset vs realised CPI inflation (2012-26)\n", T3.round(2))
