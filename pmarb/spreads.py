"""Muestreo de spreads reales en mercados de fútbol vivos para calibrar el supuesto de medio spread.
Lanza `sample` varias veces (p. ej. cada hora durante una o dos semanas) y luego `summary`."""
import json
import os
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from . import polymarket as pm

MAX_HOURS = 48
HOUR_BUCKETS = [(0, 1), (1, 6), (6, 24), (24, 48)]  # horas hasta el saque
BUCKETS = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
           (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]


def sample(tag_slug="soccer", out="spreads.csv", max_markets=300, max_hours=MAX_HOURS, log=print):
    """Solo mercados con saque en (0, max_hours] horas; el tiempo restante va en la columna hours_to_kickoff."""
    now = datetime.now(timezone.utc)
    rows = []
    for ev in pm.iter_events(tag_slug, closed=False, max_pages=10):
        for m in ev.get("markets", []):
            mk = pm.parse_market(m)
            if not mk["yes_token"] or not mk["start"]:
                continue
            ko = pd.Timestamp(mk["start"])
            ko = ko.tz_convert("UTC") if ko.tzinfo else ko.tz_localize("UTC")
            htk = (ko - now).total_seconds() / 3600
            if not 0 < htk <= max_hours:
                continue
            try:
                b = pm.book_snapshot(mk["yes_token"])
                s = pm.spread(mk["yes_token"])
            except Exception as e:  # mercado sin libro o error puntual
                log(f"skip {mk['id']}: {e}")
                continue
            if not b:
                continue
            rows.append({"time": now.isoformat(), "market": mk["id"], "token": mk["yes_token"],
                         "hours_to_kickoff": round(htk, 2), **b, "spread_endpoint": s, "half_spread": s / 2})
            if len(rows) >= max_markets:
                break
        if len(rows) >= max_markets:
            break
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(out, mode="a", header=not os.path.exists(out), index=False)
    log(f"{len(df)} libros muestreados -> {out}")
    return df


def summary(path="spreads.csv", out="spread_table.json", q=0.5, min_n=5):
    """Distribución del MEDIO spread por tramo de precio (mid) x horas al saque.
    Tabla [(p_lo, p_hi, h_lo, h_hi, half)] con el cuantil q para el backtest; celdas con < min_n
    observaciones se omiten (el backtest usa entonces el peor medio spread de la tabla)."""
    df = pd.read_csv(path)
    df = df[df.hours_to_kickoff.between(0, MAX_HOURS, inclusive="right")]
    df["p_bucket"] = pd.cut(df.mid, [b[0] for b in BUCKETS] + [1.0], right=False)
    df["h_bucket"] = pd.cut(df.hours_to_kickoff, [h[0] for h in HOUR_BUCKETS] + [MAX_HOURS], right=False)
    g = df.groupby(["h_bucket", "p_bucket"], observed=True).half_spread
    st = g.describe(percentiles=[0.25, 0.5, 0.75, 0.9])
    table = [(float(p.left), float(p.right), float(h.left), float(h.right), float(x.quantile(q)))
             for (h, p), x in g if len(x) >= min_n]
    json.dump({"quantile": q, "n": len(df), "columns": ["p_lo", "p_hi", "h_lo", "h_hi", "half"],
               "table": table}, open(out, "w"), indent=2)
    return st, table
