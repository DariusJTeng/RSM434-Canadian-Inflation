"""Write the JSON files the website reads (docs/data) and copy charts (docs/img).
Only DERIVED results are written - no raw vendor prices or index levels."""
import json, shutil, pathlib, contextlib, io
import pandas as pd, numpy as np
with contextlib.redirect_stdout(io.StringIO()):          # these modules print when imported
    import hedge as h, drivers as dr, proxy_tests as pt, subcomponent_tests as st

DOCS = pathlib.Path("docs"); (DOCS / "data").mkdir(parents=True, exist_ok=True); (DOCS / "img").mkdir(exist_ok=True)
L = h.LIAB
KEYS = {"Cash only": "cash", "Nominal GoC (dur-matched)": "nominal", "RRB dur-matched (brief)": "rrb",
        "RRB longest + cash": "rrb_long", "RRB barbell long+short": "rrb_barbell",
        "Proxy: rates only": "proxy_rates", "Proxy: rates+commod": "proxy_commod"}

def dump(name, rows):
    rows = json.loads(pd.DataFrame(rows).replace({np.nan: None}).to_json(orient="records", date_format="iso"))
    (DOCS / "data" / f"{name}.json").write_text(json.dumps(rows))

# monthly paths: liability vs hedges, gap, rolling 12m tracking error
rl = L["L"].pct_change()
paths = pd.DataFrame({"date": L.index.strftime("%Y-%m-%d"), "liability": L["L"].round(3).values})
for name, key in KEYS.items():
    r = h.H[name].reindex(L.index)
    v = L["L"].iloc[0] * (1 + r.fillna(0)).cumprod()
    paths[key] = v.round(3).values
    paths[f"gap_{key}"] = (v - L["L"]).round(3).values
    paths[f"te12_{key}"] = ((r - rl).rolling(12).std() * np.sqrt(12) * 100).round(2).values
dump("paths", paths)

S = h.H.apply(h.report).T
summ = [{"hedge": n, "key": KEYS[n], "te_pct": round(S.at[n, "TE % of L"], 2), "te_musd": round(S.at[n, "TE $M/yr"], 2),
         "te_2021_22": round(S.at[n, "TE % 2021-2022"], 2), "worst_month_musd": round(S.at[n, "worst month $M"], 2),
         "shortfall_musd": round(S.at[n, "Shortfall end $M"], 2),
         "shortfall_pct": round(S.at[n, "Shortfall end $M"] / L["L"].iloc[-1] * 100, 2)} for n in KEYS]
dump("summary", summ)
s = {r["key"]: r for r in summ}

# rolling start dates
roll = pd.read_csv("out/rolling_results.csv", parse_dates=["start"])
rk = {"Cash only": "cash", "RRB dur-matched": "rrb", "Proxy: rates only": "proxy_rates", "Proxy: rates+commod": "proxy_commod"}
sf = roll.pivot_table(index="start", columns="hedge", values="shortfall % of L").rename(columns=rk)
te = roll.pivot_table(index="start", columns="hedge", values="TE %").rename(columns=rk)
rr = sf.round(2).reset_index(); rr["start"] = rr["start"].dt.strftime("%Y-%m-%d")
dump("rolling", rr)
gap = sf["proxy_commod"] - sf["proxy_rates"]
rstats = {k: {"mean": round(sf[k].mean(), 2), "worst": round(sf[k].min(), 2), "best": round(sf[k].max(), 2),
              "te": round(te[k].mean(), 2)} for k in sf.columns}

# weights
lab = {"cash": "CAD T-bills", "nom_10y": "Nominal GoC 10y", "nom_long": "Nominal GoC long", "tips_hdg": "US TIPS (CAD-hedged)",
       "energy": "BCOM Petroleum", "agri": "BCOM Agriculture", "livestock": "BCOM Livestock", "tsx_energy": "TSX Energy",
       "tsx_re": "TSX Real Estate"}
dump("weights", [{"asset": lab[a], "rates_commod_pct": round(h.w_proxy[a] * 100, 1),
                  "rates_only_pct": round(h.w_rates[a] * 100, 1)} for a in h.R.columns])

# CPI drivers (step 1)
A = dr.A.reset_index().rename(columns={"index": "component"})
A.columns = ["component", "weight_pct", "vol_pct", "var_share_pct", "cum_share_pct", "cum_share_2123_pct"]
dump("drivers", A)

# proxy tests (computed, not typed)
def pick(T, key, piece, proxy, verdict):
    r = T.loc[key]; return {"piece": piece, "proxy": proxy, "r2_in": round(float(r["ΔR2 in-sample"]), 2),
                            "r2_oos": round(float(r["R2 out-of-sample"]), 2), "verdict": verdict}
P, U = pt.T, st.T
tests = [pick(P, "gasoline CPI <- BCOM Petroleum TR (lags 0-6)", "Gasoline", "BCOM Petroleum", "Passes"),
         pick(U, "fuel_oil <- BCOM Petroleum TR", "Fuel oil", "BCOM Petroleum", "Passes"),
         pick(P, "energy CPI <- TSX Energy TR (lags 0-6)", "Energy (all)", "TSX Energy equities", "Passes"),
         pick(U, "mic <- GoC 5y yield (chg)", "Mortgage interest cost", "GoC 5y yield", "Passes with 1-year lag"),
         pick(U, "replacement <- TSX Real Estate TR", "Replacement cost", "TSX Real Estate", "Weak"),
         pick(U, "rent <- TSX Real Estate TR", "Rent", "TSX Real Estate", "Fails"),
         pick(P, "shelter CPI <- TSX Real Estate TR (lags 0-12)", "Shelter (all)", "TSX Real Estate", "Fails"),
         pick(U, "natgas_cpi <- BCOM Natural Gas TR", "Natural gas", "BCOM Natural Gas", "Fails"),
         pick(U, "electricity <- BCOM Natural Gas TR", "Electricity", "BCOM Natural Gas", "Fails"),
         pick(U, "meat <- BCOM Livestock TR", "Meat", "BCOM Livestock", "Fails"),
         pick(U, "bakery <- BCOM Grains TR", "Bakery & cereals", "BCOM Grains", "Fails"),
         pick(P, "food CPI <- BCOM Agriculture TR (lags 0-6)", "Food (all)", "BCOM Agriculture", "Fails")]
dump("proxytests", tests)

# basket by hedgeability (2025 weights)
b = pd.read_csv("data/clean/cpi_basket_weights.csv", index_col=0)["2025"]
petro, mic, repl = b["Gasoline"] + b["Fuel oil and other fuels"], b["Mortgage interest cost"], b["Homeowners' replacement cost"]
basket = [{"group": "Hedgeable now", "detail": "Gasoline, fuel oil (petroleum futures)", "weight_pct": round(petro, 2)},
          {"group": "Hedgeable with ~1-year lag", "detail": "Mortgage interest cost (nominal rates)", "weight_pct": round(mic, 2)},
          {"group": "Weak link", "detail": "Homeowners' replacement cost (REITs)", "weight_pct": round(repl, 2)},
          {"group": "No proxy", "detail": "Rent, property tax, utilities, food, services, everything else",
           "weight_pct": round(100 - petro - mic - repl, 2)}]
dump("basket", basket)

# headline numbers for the page text
(DOCS / "data" / "headline.json").write_text(json.dumps({
    "as_of": f"{L.index[-1]:%Y-%m-%d}", "downloaded": "2026-11-20",
    "liability_musd": round(L["L"].iloc[-1], 2), "liability_start_musd": round(L["L"].iloc[0], 2),
    "rrb_shortfall_musd": s["rrb"]["shortfall_musd"], "proxy_rates_shortfall_musd": s["proxy_rates"]["shortfall_musd"],
    "proxy_commod_shortfall_musd": s["proxy_commod"]["shortfall_musd"],
    "cost_of_losing_rrbs_musd": round(s["rrb"]["shortfall_musd"] - s["proxy_rates"]["shortfall_musd"], 2),
    "hedgeable_now_pct": round(petro, 2), "hedgeable_incl_rates_pct": round(petro + mic, 2),
    "rolling": rstats, "rolling_windows": int(len(sf)),
    "rolling_first": f"{sf.index[0]:%Y-%m}", "rolling_last": f"{sf.index[-1]:%Y-%m}",
    "sleeve_helped_pct": round((gap > 0).mean() * 100, 0), "sleeve_worst_pct": round(gap.min(), 2),
    "sleeve_worst_start": f"{gap.idxmin():%Y-%m}", "sleeve_best_pct": round(gap.max(), 2),
    "sleeve_best_start": f"{gap.idxmax():%Y-%m}"}, indent=1))

for f in ["fig2_strong_vs_weak.png", "fig1_explanatory_power.png", "fig3_corr_with_liability.png"]:
    shutil.copy(pathlib.Path("out") / f, DOCS / "img" / f)
print("wrote", sorted(p.name for p in (DOCS / "data").iterdir()))
