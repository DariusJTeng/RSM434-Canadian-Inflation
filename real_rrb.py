"""Overwrite the synthetic RRB panel with REAL Bloomberg prices wherever they exist.
Real yield and duration are backed out of the real clean price; index ratio and coupons
follow the RRB rules (same as fake_rrb.py). Months without real data stay synthetic."""
import pandas as pd, numpy as np
from scipy.optimize import brentq
import fake_rrb as F
from load_data import bbg

FILES = {"RRB 4.25% 2026": "BGN_RRB26.xlsx", "RRB 4.00% 2031": "BGN_RRB31.xlsx", "RRB 3.00% 2036": "BGN_RRB36.xlsx",
         "RRB 2.00% 2041": "BGN_RRB41.xlsx", "RRB 1.50% 2044": "BGN_RRB44.xlsx", "RRB 1.25% 2047": "BGN_RRB47.xlsx",
         "RRB 0.50% 2050": "BGN_RRB50.xlsx", "RRB 0.25% 2054": "BGN_RRB54.xlsx"}
INFO = F.ISSUES.set_index("issue")
rrb = F.rrb.copy().set_index(["issue", "date"])

for issue, f in FILES.items():
    px = bbg(f).resample("ME").last().loc[:"2026-10-31"].dropna()
    mat, cpn = INFO.at[issue, "maturity"], INFO.at[issue, "coupon"]
    for t, clean in px.items():
        if (issue, t) not in rrb.index or t >= mat:
            continue
        y = brentq(lambda yy: F.real_prices(t, mat, cpn, yy)[1] - clean, -5, 15)   # real yield from price
        dirty, _, acc = F.real_prices(t, mat, cpn, y)
        rrb.loc[(issue, t), ["real_yield", "real_clean", "real_dirty", "accrued", "mod_dur", "synthetic"]] = \
            [y, clean, dirty, acc, F.mod_duration(t, mat, cpn, y), False]

rrb = rrb.reset_index().sort_values(["issue", "date"])
rrb["indexed_dirty"] = rrb["real_dirty"] * rrb["index_ratio"]
prev = rrb.groupby("issue")["indexed_dirty"].shift(1)
tr = (rrb["indexed_dirty"] + rrb["coupon_cash"]) / prev - 1
switch = rrb.groupby("issue")["synthetic"].shift(1).ne(rrb["synthetic"]) & rrb["synthetic"].eq(False)
rrb["total_return"] = np.where(switch, rrb["total_return"], tr)   # first real month: keep synthetic return (no splice jump)
rrb.to_csv("data/clean/rrb_prices.csv", index=False)

if __name__ == "__main__":
    print(rrb.groupby("issue")["synthetic"].agg(real_months=lambda s: (~s.astype(bool)).sum(),
                                                 synthetic_months=lambda s: s.astype(bool).sum()))
    last = rrb[rrb.date == rrb.date.max()].set_index("issue")
    print(last[["real_clean", "real_yield", "mod_dur", "synthetic"]].round(2))
