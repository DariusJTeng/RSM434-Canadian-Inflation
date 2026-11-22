# Replicating Canadian Inflation

**Can a pension or insurer hedge Canadian CPI without Real Return Bonds?**
RSM434 Industry Project 6 (with CIBC) · Darius Jonathan Teng · Rotman Commerce, University of Toronto

**Live dashboard:** https://dariusteng.org/cpi-hedge/

Ottawa stopped issuing Real Return Bonds (RRBs) in 2022. This project measures what that costs a holder of a
CPI-linked liability, and how well tradable proxies (nominal Government of Canada bonds, US TIPS, commodities,
inflation-sensitive equities) can stand in.

## Headline results (data to 31 Oct 2026)

| | |
|---|---|
| Liability | $100M × CPI growth, paid 10 years after 31 Dec 2017, discounted at the long RRB real yield |
| Duration-matched RRBs | end gap −$1.2M, tracking error 2.7% a year |
| Best rates-only proxy (nominal GoC + CAD-hedged TIPS) | end gap −$11.5M, tracking error 2.1% a year |
| **Cost of losing RRBs** | **≈ $10M per $100M of liability over 9 years** |
| CPI basket with a working tradable proxy | 4.3% (petroleum); 9.5% including mortgage interest via nominal rates |

Across 151 start dates (walk-forward, 5-year windows), RRBs end on average 3.7 points of the liability ahead of the
rates-only proxy. A small commodity/equity sleeve narrows the gap in most windows, but as a return premium that
raises tracking error and reverses in commodity crashes (−3.2 points for a Mar-2015 start).

## Repository layout

```
pipeline/
  load_data.py           all raw files -> one month-end panel (data/clean/monthly.csv)
  fake_rrb.py            modelled prices for the 8 RRBs (flat-breakeven curve), used where real prices are missing
  real_rrb.py            overwrites them with Bloomberg prices; backs out real yield and duration
  hedge.py               liability, RRB benchmark and variants, proxy hedge optimisation, out-of-sample test
  rolling.py             walk-forward start-date test (151 windows)
  drivers.py             step 1: which CPI components drive variance and level; proxy regressions
  proxy_tests.py         CPI components vs tradable instruments (distributed lags, HAC SEs, out-of-sample R²)
  subcomponent_tests.py  shelter and food sub-components vs their best proxy
  hedgeable_share.py     share of the CPI basket with a working proxy (official basket weights)
  evidence.py            regression tables and charts
  export_site.py         writes the derived JSON the website reads (docs/data) and copies charts (docs/img)
run_all.py               runs everything in order
docs/                    the website (GitHub Pages serves this folder)
data/                    public downloads (StatCan, Bank of Canada, FRED)
data/vendor/             Bloomberg / Capital IQ exports - NOT in this repository
```

## Reproducing

```bash
pip install -r requirements.txt
python run_all.py
cd docs && python -m http.server 8000      # then open http://localhost:8000
```

`run_all.py` needs the vendor files in `data/vendor/`, which are licensed and not published here:

| File | Source | Series |
|---|---|---|
| `BGN_RRB26.xlsx` … `BGN_RRB54.xlsx` | Bloomberg | month-end `PX_LAST` of the 8 Government of Canada RRBs |
| `BCOMENTR`, `BCOMPETR`, `BCOMAGTR`, `BCOMLITR_Index`, `2_BCOMGRTR`, `2_BCOMNGTR` (.xlsx) | Bloomberg | BCOM total return sub-indices, in CAD |
| `LBUTTRUU.xlsx` | Bloomberg | US TIPS total return index |
| `CANADA_3_MONTH_GOVERNMENT_BOND.xlsx` | Bloomberg | Canada 3-month T-bill yield (GTCAD3MO) |
| `CAD_SWAP_OIS_1Y/5Y/10Y.xlsx` | Bloomberg | CAD OIS swap rates |
| `TTENAR_...xlsx` | Bloomberg | S&P/TSX Capped Energy total return |
| `SPTSX_Capped_Real_Estate_Index_TTRE.xls`, `Dow_Jones_*.xlsx`, `SP_GSCI_*.xlsx` | S&P Capital IQ | TSX Real Estate TR; legacy BCOM and GSCI series |

Public files (included in `data/`): StatCan 18-10-0004-01 (8 major components), Bank of Canada selected bond yields,
FRED `DTB3`, `DFII10`, `T10YIE`, `DEXCAUS`, `DCOILWTICO`, `PFOODINDEXM`, `CPIAUCNS`.
Re-download the two large StatCan tables (full CSV) into `data/`:
`18100004_full.csv` (table 18-10-0004-01, all products) and `18100007_weights.csv` (table 18-10-0007-01, basket weights).

## Method notes

- CPI enters the liability with its publication lag: at each month-end, only the previous month's CPI is known.
- RRB reference CPI: CPI three months back, interpolated through the month.
- Proxy weights minimise in-sample (2004–2017) tracking error against a constant 10-year liability, fully funded,
  shorting allowed only in nominal GoC. Out of sample, rate positions scale with the liability's duration.
- RRB prices before Oct 2015 are modelled (validated against Bloomberg on the overlap, monthly price-change
  correlation 0.83–0.99) and are used only in the start-date test.

Not investment advice. Vendor data remain the property of their providers; this repository publishes only derived results.
