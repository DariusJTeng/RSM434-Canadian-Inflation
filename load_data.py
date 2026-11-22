"""Load every raw file in data/ into one clean month-end panel: data/clean/monthly.csv"""
import pandas as pd, numpy as np
from pathlib import Path

D = Path("data")                 # public downloads (StatCan, Bank of Canada, FRED)
VEND = D / "vendor"              # Bloomberg / CapIQ / FactSet exports - NOT published (git-ignored)
END = "2026-10-31"        # last COMPLETE month; update when you refresh the data

def fred(name):                        # FRED csv: observation_date, value (daily or monthly)
    s = pd.read_csv(D / f"{name}.csv", index_col=0, parse_dates=True).iloc[:, 0]
    return pd.to_numeric(s, errors="coerce")

def capiq(file, sheet="1 - Chart Data"):   # CapIQ export: date column + one value column
    x = pd.read_excel(VEND / file, sheet_name=sheet, header=None)
    dates = pd.to_datetime(x[0], errors="coerce", format="mixed")
    s = pd.Series(pd.to_numeric(x[1], errors="coerce").values, index=dates)
    return s[s.index.notna()].dropna().sort_index()

def boc_bonds():                       # Bank of Canada "Selected bond yields" csv
    lines = open(D / "bond_yields_all.csv", encoding="utf-8-sig").read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith('"date"'))
    b = pd.read_csv(D / "bond_yields_all.csv", skiprows=start, encoding="utf-8-sig",
                    index_col=0, parse_dates=True)
    b = b.apply(pd.to_numeric, errors="coerce")
    return b.rename(columns={"BD.CDN.2YR.DQ.YLD": "goc_2y", "BD.CDN.5YR.DQ.YLD": "goc_5y",
                             "BD.CDN.10YR.DQ.YLD": "goc_10y", "BD.CDN.LONG.DQ.YLD": "goc_long",
                             "BD.CDN.RRB.DQ.YLD": "rrb_long"})[
        ["goc_2y", "goc_5y", "goc_10y", "goc_long", "rrb_long"]]

def boc_tbill():                       # Bank of Canada V39065 csv (has a summary block on top)
    lines = open(D / "V39065_Treasury_bills_3_month.csv").read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("Date,"))
    t = pd.read_csv(D / "V39065_Treasury_bills_3_month.csv", skiprows=start,
                    index_col=0, parse_dates=True).iloc[:, 0]
    return pd.to_numeric(t, errors="coerce")      # "Bank holiday" -> NaN

def statcan_cpi():                     # StatCan 18-10-0004-01, wide format (one row per product)
    raw = pd.read_csv(D / "1810000401-eng.csv", skiprows=9, header=0, encoding="utf-8-sig")
    raw = raw.rename(columns={raw.columns[0]: "item"}).dropna(subset=["item"])
    raw["item"] = raw["item"].str.replace(r"\s\d+$", "", regex=True).str.strip()  # drop footnote nos.
    raw = raw[raw.iloc[:, 1].notna() & raw["item"].ne("")]
    names = {"All-items": "headline", "Food": "food", "Shelter": "shelter",
             "Household operations, furnishings and equipment": "household",
             "Clothing and footwear": "clothing", "Transportation": "transport",
             "Gasoline": "gasoline", "Health and personal care": "health",
             "Recreation, education and reading": "recreation",
             "Alcoholic beverages, tobacco products and recreational cannabis": "alc_tob",
             "All-items excluding food and energy": "core_xfe", "Energy": "energy",
             "Goods": "goods", "Services": "services"}
    raw = raw[raw["item"].isin(names)].set_index("item")
    cpi = raw.T.apply(pd.to_numeric, errors="coerce").rename(columns=names)
    cpi.index = pd.to_datetime(cpi.index, format="%B %Y")
    return cpi.add_prefix("cpi_")

def bbg(file):                         # Bloomberg export: metadata rows, then "Date | PX_LAST | ..."
    d = pd.read_excel(VEND / file, header=None)
    i = d.index[d[0].astype(str).eq("Date")][0]
    s = pd.Series(pd.to_numeric(d.iloc[i + 1:, 1], errors="coerce").values,
                  index=pd.to_datetime(d.iloc[i + 1:, 0], errors="coerce"))
    s = s[s.index.notna()].dropna().sort_index()
    if s.index.nunique() < 10:
        raise ValueError(f"{file}: export looks frozen (one date repeated) - pull again")
    return s[~s.index.duplicated()]

def to_month_end(s, how="last"):
    s = s.dropna()
    return s.resample("ME").last() if how == "last" else s.resample("ME").mean()

def build():
    # --- CPI: index by the month it MEASURES (month-end stamp) ---
    cpi = statcan_cpi()
    cpi.index = cpi.index + pd.offsets.MonthEnd(0)
    us = fred("CPIAUCNS").interpolate(limit=1)        # Oct-2025 missing (US shutdown) -> interpolated
    us.index = us.index + pd.offsets.MonthEnd(0)

    # --- market data: month-end close ---
    ix = {
        "bcom_energy_tr": capiq("Dow_Jones-UBS_Energy_Total_Return_Subindex.xlsx"),
        "bcom_agri_tr":   capiq("Dow_Jones_AIG_Agriculture_Sub_Total_Return_Index.xlsx"),
        "bcom_live_tr":   capiq("Dow_Jones_AIG_Livestock_Total_Return_Index.xlsx").loc["2004-03-08":],  # rebasing break before this
        "gsci_agliv_tr":  capiq("SP_GSCI_Agriculture__Livestock_Total_Return_Index.xlsx"),
        "tsx_re_tr":      capiq("SPTSX_Capped_Real_Estate_Index_TTRE.xls", sheet="Pane 1"),
    }
    # Bloomberg (pulled in CAD where noted -> do NOT convert again)
    ix.update({
        "bbg_energy_tr_cad":    bbg("BCOMENTR.xlsx"),
        "bbg_petroleum_tr_cad": bbg("BCOMPETR.xlsx"),
        "bbg_agri_tr_cad":      bbg("BCOMAGTR.xlsx"),
        "bbg_live_tr_cad":      bbg("BCOMLITR_Index.xlsx"),
        "tsx_energy_tr":        bbg("TTENAR_SP500_TSX_ENERGY_CAPPED_TOTAL_RETURN.xlsx"),
        "us_tips_tr_usd":       bbg("LBUTTRUU.xlsx"),
        "cad_ois_1y":           bbg("CAD_SWAP_OIS_1Y.xlsx"),
        "cad_ois_5y":           bbg("CAD_SWAP_5Y.xlsx"),
        "cad_ois_10y":          bbg("CAD_SWAP_10Y.xlsx"),
        "bbg_grains_tr_cad":    bbg("2_BCOMGRTR.xlsx"),
        "bbg_natgas_tr_cad":    bbg("2_BCOMNGTR.xlsx"),
        "ca_tbill_bbg":         bbg("CANADA_3_MONTH_GOVERNMENT_BOND.xlsx"),   # GTCAD3MO, monthly from 2000
    })
    mkt = pd.DataFrame({k: to_month_end(v) for k, v in ix.items()})
    mkt = mkt.join(boc_bonds().resample("ME").last())
    mkt["ca_tbill_3m"] = to_month_end(boc_tbill())
    for code, name in [("DTB3", "us_tbill_3m"), ("DFII10", "us_tips10_real"),
                       ("T10YIE", "us_be10"), ("DEXCAUS", "usdcad")]:
        mkt[name] = to_month_end(fred(code))
    mkt["wti_usd"] = to_month_end(fred("DCOILWTICO"), how="mean")   # monthly average for pass-through
    food = fred("PFOODINDEXM"); food.index = food.index + pd.offsets.MonthEnd(0)
    mkt["imf_food_usd"] = food

    panel = cpi.join(us.rename("us_cpi"), how="outer").join(mkt, how="outer")
    panel = panel.loc["2004-01-31":END]          # drop the partial current month
    Path("data/clean").mkdir(exist_ok=True)
    panel.to_csv("data/clean/monthly.csv")
    return panel

if __name__ == "__main__":
    p = build()
    cov = pd.DataFrame({"first": p.apply(lambda s: s.first_valid_index()),
                        "last": p.apply(lambda s: s.last_valid_index()),
                        "n_obs": p.notna().sum()})
    print(cov.to_string())
