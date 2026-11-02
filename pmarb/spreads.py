"""Muestreo de spreads reales en mercados de fútbol vivos para calibrar el supuesto de medio spread.
Lanza `sample` varias veces (p. ej. cada hora durante una o dos semanas) y luego `summary`."""
import json
import os
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from . import polymarket as pm

BUCKETS = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
           (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]


def sample(tag_slug="soccer", out="spreads.csv", max_markets=300, log=print):
    now = datetime.now(timezone.utc)
    rows = []
    for ev in pm.iter_events(tag_slug, closed=False, max_pages=10):
        for m in ev.get("markets", []):
            mk = pm.parse_market(m)
            if not mk["yes_token"]:
                continue
            try:
                b = pm.book_snapshot(mk["yes_token"])
                s = pm.spread(mk["yes_token"])
            except Exception as e:  # mercado sin libro o error puntual
                log(f"skip {mk['id']}: {e}")
                continue
            if not b:
                continue
            ko = pd.Timestamp(mk["start"]) if mk["start"] else None
            rows.append({"time": now.isoformat(), "market": mk["id"], "token": mk["yes_token"],
                         "hours_to_kickoff": (ko.tz_convert("UTC") - now).total_seconds() / 3600 if ko is not None and ko.tzinfo else None,
                         **b, "spread_endpoint": s, "half_spread": s / 2})
            if len(rows) >= max_markets:
                break
        if len(rows) >= max_markets:
            break
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(out, mode="a", header=not os.path.exists(out), index=False)
    log(f"{len(df)} libros muestreados -> {out}")
    return df


def summary(path="spreads.csv", out="spread_table.json", q=0.5, max_hours=24):
    """Distribución del MEDIO spread por tramo de precio (mid). Tabla con el cuantil q para el backtest."""
    df = pd.read_csv(path)
    if max_hours is not None and "hours_to_kickoff" in df:
        df = df[(df.hours_to_kickoff.isna()) | (df.hours_to_kickoff.between(0, max_hours))]
    df["bucket"] = pd.cut(df.mid, [b[0] for b in BUCKETS] + [1.0], right=False)
    st = df.groupby("bucket", observed=True).half_spread.describe(percentiles=[0.25, 0.5, 0.75, 0.9])
    table = [(float(iv.left), float(iv.right), float(df[df.bucket == iv].half_spread.quantile(q)))
             for iv in st.index]
    json.dump({"quantile": q, "n": len(df), "table": table}, open(out, "w"), indent=2)
    return st, table
