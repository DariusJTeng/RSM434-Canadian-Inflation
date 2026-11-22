"""How much of the CPI basket has a tradable proxy? Uses official basket weights (18-10-0007-01)."""
import pandas as pd
d = pd.read_csv("data/18100007_weights.csv", low_memory=False)
g = d[(d.GEO == "Canada") & (d["Price period of weight"] == "Weight at basket link month prices")
      & (d["Geographic distribution of weight"] == "Distribution to selected geographies")]
w = g.pivot_table(index="Products and product groups", columns="REF_DATE", values="VALUE")
w.to_csv("data/clean/cpi_basket_weights.csv")

CLASS = {   # verdicts from proxy_tests.py / subcomponent_tests.py (out-of-sample 2018-26)
    "Gasoline": "1 Hedgeable now (petroleum futures)",
    "Fuel oil and other fuels": "1 Hedgeable now (petroleum futures)",
    "Mortgage interest cost": "2 Hedgeable with ~1y lag (nominal rates)",
    "Homeowners' replacement cost": "3 Weak link (REITs)",
}
yr = w.columns.max()
basket = w[yr]
rows = [(item, basket[item], cls) for item, cls in CLASS.items()]
t = pd.DataFrame(rows, columns=["item", f"weight {yr} %", "class"])
summary = t.groupby("class")[f"weight {yr} %"].sum()
summary["4 No proxy (everything else)"] = 100 - summary.sum()
print(t.to_string(index=False), "\n")
print(summary.round(2).to_string())
print("\nCumulative: hedgeable now %.1f%% | incl. lagged rates %.1f%% | incl. weak links %.1f%%"
      % tuple(summary.cumsum().iloc[:3]))
