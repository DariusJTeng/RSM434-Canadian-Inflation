"""Synthetic ("fake") prices for the 8 remaining Government of Canada RRBs.
Placeholder until real prices arrive - same output format, so the rest of the
pipeline won't change when you swap in the real file.

Assumption: FLAT BREAKEVEN CURVE.
    real_yield(maturity) = nominal GoC yield at that maturity - (goc_long - rrb_long)
"""
import pandas as pd, numpy as np

p = pd.read_csv("data/clean/monthly.csv", index_col=0, parse_dates=True)

# coupon, maturity, approx. first issue date (CHECK against real data)
ISSUES = pd.DataFrame([
    ("RRB 4.25% 2026", 4.25, "2026-12-01", "1995-12-01"),
    ("RRB 4.00% 2031", 4.00, "2031-12-01", "1999-12-01"),
    ("RRB 3.00% 2036", 3.00, "2036-12-01", "2003-12-01"),
    ("RRB 2.00% 2041", 2.00, "2041-12-01", "2008-06-01"),
    ("RRB 1.50% 2044", 1.50, "2044-12-01", "2010-12-01"),
    ("RRB 1.25% 2047", 1.25, "2047-12-01", "2013-12-01"),
    ("RRB 0.50% 2050", 0.50, "2050-12-01", "2016-12-01"),
    ("RRB 0.25% 2054", 0.25, "2054-12-01", "2021-06-01"),
], columns=["issue", "coupon", "maturity", "first_issue"])
ISSUES[["maturity", "first_issue"]] = ISSUES[["maturity", "first_issue"]].apply(pd.to_datetime)
NOISE_BP = 0.0      # set e.g. 3.0 to add issue-specific yield noise (bp, i.i.d. monthly)
SEED = 434

# ---------- 1. real yield curve each month ----------
NODES = {2: "goc_2y", 5: "goc_5y", 10: "goc_10y", 30: "goc_long"}   # long benchmark ~30y
be = p["goc_long"] - p["rrb_long"]                                   # flat breakeven

def real_yield(date, years):
    nom = np.interp(years, list(NODES), [p.at[date, c] for c in NODES.values()])
    return nom - be[date]                                            # % per year

# ---------- 2. reference CPI and index ratio ----------
cpi = p["cpi_headline"]                      # indexed by the month the CPI MEASURES

def ref_cpi(date):
    """Canadian RRB reference CPI: CPI 3 months back, interpolated through the month."""
    m3 = (date - pd.offsets.MonthEnd(3))      # month-end of month m-3
    m2 = (date - pd.offsets.MonthEnd(2))
    if m3 not in cpi.index or m2 not in cpi.index or pd.isna(cpi[m3]) or pd.isna(cpi[m2]):
        return np.nan
    d, D = date.day, date.days_in_month
    return cpi[m3] + (d - 1) / D * (cpi[m2] - cpi[m3])

# ---------- 3. bond maths (semi-annual coupons on Jun 1 / Dec 1) ----------
def coupon_dates(settle, maturity):
    out, d = [], maturity
    while d > settle:
        out.append(d)
        d = d - pd.DateOffset(months=6)
    return sorted(out), d                     # future coupons, last coupon date

def real_prices(settle, maturity, cpn, y):
    """Real (unindexed) dirty, clean, accrued per 100 face at yield y (%)."""
    cfs, last = coupon_dates(settle, maturity)
    nxt = cfs[0]
    frac = (nxt - settle).days / (nxt - last).days            # fraction of period to next coupon
    k = np.arange(len(cfs)) + frac                            # periods to each cash flow
    cf = np.full(len(cfs), cpn / 2); cf[-1] += 100
    dirty = (cf / (1 + y / 200) ** k).sum()
    accrued = cpn / 2 * (1 - frac)
    return dirty, dirty - accrued, accrued

def mod_duration(settle, maturity, cpn, y, h=0.01):
    up = real_prices(settle, maturity, cpn, y + h)[0]
    dn = real_prices(settle, maturity, cpn, y - h)[0]
    mid = real_prices(settle, maturity, cpn, y)[0]
    return (dn - up) / (2 * mid * h / 100)

# ---------- 4. build the panel ----------
rng = np.random.default_rng(SEED)
dates = p.index[p["rrb_long"].notna() & p["goc_2y"].notna()]
rows = []
for _, b in ISSUES.iterrows():
    for t in dates:
        if t < b.first_issue or t >= b.maturity:
            continue
        yrs = (b.maturity - t).days / 365.25
        y = real_yield(t, yrs) + NOISE_BP / 100 * rng.standard_normal()
        dirty, clean, acc = real_prices(t, b.maturity, b.coupon, y)
        rows.append(dict(date=t, issue=b.issue, years_to_mat=yrs, real_yield=y,
                         real_clean=clean, real_dirty=dirty, accrued=acc,
                         ref_cpi=ref_cpi(t),
                         mod_dur=mod_duration(t, b.maturity, b.coupon, y)))
rrb = pd.DataFrame(rows).sort_values(["issue", "date"])

# index ratio: TRUE base CPI unknown here -> normalise to 1 at each issue's first obs.
# (Returns don't depend on the base; replace with real index ratios later.)
rrb = rrb.dropna(subset=["ref_cpi"])
rrb["index_ratio"] = rrb["ref_cpi"] / rrb.groupby("issue")["ref_cpi"].transform("first")
rrb["indexed_dirty"] = rrb["real_dirty"] * rrb["index_ratio"]       # what you actually pay

# coupons paid during the month (Jun 1 / Dec 1), indexed at the coupon date
def coupon_cash(r):
    if r.date.month not in (6, 12):
        return 0.0
    cpn = ISSUES.set_index("issue").at[r.issue, "coupon"]
    pay_date = r.date.replace(day=1)
    return cpn / 2 * ref_cpi(pay_date) / (r.ref_cpi / r.index_ratio)   # same base as index_ratio
rrb["coupon_cash"] = rrb.apply(coupon_cash, axis=1)

prev = rrb.groupby("issue")["indexed_dirty"].shift(1)
rrb["total_return"] = (rrb["indexed_dirty"] + rrb["coupon_cash"]) / prev - 1

rrb["synthetic"] = True                 # flag: replace with real prices when available
rrb.to_csv("data/clean/rrb_prices.csv", index=False)

if __name__ == "__main__":
    last = rrb[rrb.date == rrb.date.max()].set_index("issue")
    print("Snapshot", rrb.date.max().date())
    print(last[["years_to_mat", "real_yield", "real_clean", "index_ratio", "mod_dur"]].round(2))
    tr = rrb.pivot(index="date", columns="issue", values="total_return")
    print("\nAnnualised return / vol by issue (%):")
    print(pd.DataFrame({"ret": tr.mean() * 1200, "vol": tr.std() * np.sqrt(12) * 100,
                        "2022 return": tr.loc["2022"].add(1).prod() * 100 - 100}).round(1))
