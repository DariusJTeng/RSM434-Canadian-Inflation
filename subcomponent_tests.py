"""Test shelter and food SUB-components against their best tradable proxy."""
import pandas as pd, numpy as np
import statsmodels.api as sm
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)

# ---------- 1. pull sub-components from the full StatCan table ----------
ITEMS = {"Rent": "rent", "Mortgage interest cost": "mic", "Homeowners' replacement cost": "replacement",
         "Property taxes and other special charges": "prop_tax", "Electricity": "electricity",
         "Natural gas": "natgas_cpi", "Fuel oil and other fuels": "fuel_oil",
         "Meat": "meat", "Dairy products and eggs": "dairy",
         "Bakery and cereal products (excluding baby food)": "bakery",
         "Fresh vegetables": "fresh_veg", "Food purchased from stores": "food_stores",
         "Food purchased from restaurants": "restaurants", "Shelter": "shelter", "Food": "food"}
raw = pd.read_csv("data/18100004_full.csv", low_memory=False,
                  usecols=["REF_DATE", "GEO", "Products and product groups", "UOM", "VALUE"])
raw = raw[(raw.GEO == "Canada") & (raw.UOM == "2002=100") & raw["Products and product groups"].isin(ITEMS)]
sub = raw.pivot_table(index="REF_DATE", columns="Products and product groups", values="VALUE").rename(columns=ITEMS)
sub.index = pd.to_datetime(sub.index) + pd.offsets.MonthEnd(0)
sub = sub.loc["2003-01-31":"2026-10-31"]
sub.add_prefix("cpi_").to_csv("data/clean/cpi_subcomponents.csv")

p = pd.read_csv("data/clean/monthly.csv", index_col=0, parse_dates=True)
dl = np.log(sub).diff()
IS_END = "2017-12-31"
inst = pd.DataFrame({
    "GoC 5y yield (chg)":    p["goc_5y"].diff() / 100,
    "TSX Real Estate TR":    np.log(p["tsx_re_tr"]).diff(),
    "BCOM Natural Gas TR":   np.log(p["bbg_natgas_tr_cad"]).diff(),
    "BCOM Petroleum TR":     np.log(p["bbg_petroleum_tr_cad"]).diff(),
    "BCOM Livestock TR":     np.log(p["bbg_live_tr_cad"]).diff(),
    "BCOM Grains TR":        np.log(p["bbg_grains_tr_cad"]).diff(),
    "BCOM Agriculture TR":   np.log(p["bbg_agri_tr_cad"]).diff(),
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
    return {"pass_through": float(pt.effect[0]), "t": float(pt.tvalue[0][0]), "F p-value": float(F.pvalue),
            "ΔR2 in-sample": fit.rsquared - seas.rsquared,
            "R2 out-of-sample": 1 - ((oos["y"] - pred) ** 2).sum() / ((oos["y"] - base) ** 2).sum()}

TESTS = [  # (component, proxy, lags)
    ("mic", "GoC 5y yield (chg)", 12), ("rent", "TSX Real Estate TR", 12),
    ("replacement", "TSX Real Estate TR", 12), ("replacement", "GoC 5y yield (chg)", 12),
    ("prop_tax", "GoC 5y yield (chg)", 12),
    ("natgas_cpi", "BCOM Natural Gas TR", 6), ("electricity", "BCOM Natural Gas TR", 6),
    ("fuel_oil", "BCOM Petroleum TR", 6),
    ("meat", "BCOM Livestock TR", 6), ("bakery", "BCOM Grains TR", 12),
    ("dairy", "BCOM Livestock TR", 6), ("fresh_veg", "BCOM Agriculture TR", 6),
    ("food_stores", "BCOM Agriculture TR", 6), ("restaurants", "BCOM Agriculture TR", 6),
]
T = pd.DataFrame({f"{c} <- {x}": lag_reg(dl[c], inst[x], k) for c, x, k in TESTS}).T

# 12-month horizon for the slow shelter pieces: does last year's rate move predict this year's inflation?
yoy = np.log(sub).diff(12)
H = pd.DataFrame({c: {"corr with GoC 5y chg, same 12m": yoy[c].corr(p["goc_5y"].diff(12)),
                      "corr with GoC 5y chg, 12m earlier": yoy[c].corr(p["goc_5y"].diff(12).shift(12))}
                  for c in ["mic", "rent", "replacement", "prop_tax", "shelter"]}).T

# how much of shelter / food MOVEMENT each piece accounts for (using index changes, no basket weights needed)
def share_of_cum(parent, kids):
    ch = {k: (sub[k].loc["2026-10-31"] / sub[k].loc["2004-01-31"] - 1) * 100 for k in kids + [parent]}
    return pd.Series(ch).round(1)

if __name__ == "__main__":
    print("A. Sub-component proxy tests (in-sample 2004-17, OOS 2018-26)\n", T.round(3), "\n")
    print("B. Shelter pieces vs interest rates at a 12-month horizon (2005-2026)\n", H.round(2), "\n")
    print("C. Cumulative price change Jan-2004 -> Oct-2026 (%)")
    print(share_of_cum("shelter", ["mic", "rent", "replacement", "prop_tax", "electricity", "natgas_cpi"]).to_string())
    print(share_of_cum("food", ["meat", "dairy", "bakery", "fresh_veg", "food_stores", "restaurants"]).to_string())
