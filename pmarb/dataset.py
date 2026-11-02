"""Dataset histórico: football-data (Pinnacle apertura + cierre) x Polymarket (precio pre-partido).
Guarda cuotas crudas: el de-vig se aplica en el backtest para poder comparar métodos."""
from datetime import datetime, time
import numpy as np
import pandas as pd
from . import polymarket as pm
from .matching import match_event, outcome_of

ODDS = ["oH", "oD", "oA", "ooH", "ooD", "ooA"]


def build(fd: pd.DataFrame, tag_slug="soccer", lead_min=10, kickoff_utc_hour=12, log=print):
    by_day = {}
    for e in pm.iter_events(tag_slug, closed=True):
        by_day.setdefault((e.get("startDate") or e.get("endDate") or "")[:10], []).append(e)
    rows = []
    for k, r in fd.iterrows():
        cands = sum((by_day.get(str(pd.Timestamp(r.date) + pd.Timedelta(days=o))[:10], []) for o in (-1, 0, 1)), [])
        ev = match_event(r.HomeTeam, r.AwayTeam, cands)
        if not ev:
            continue
        for m in ev.get("markets", []):
            mk = pm.parse_market(m)
            if not mk["yes_token"]:
                continue
            ko = pd.Timestamp(mk["start"]) if mk["start"] else pd.Timestamp(datetime.combine(r.date, time(kickoff_utc_hour)))
            ko = ko.tz_convert("UTC") if ko.tzinfo else ko.tz_localize("UTC")
            ts = int(ko.timestamp()) - lead_min * 60
            p = pm.price_at(mk["yes_token"], ts)
            if p is None or not 0 < p < 1:
                continue
            o = outcome_of(mk["question"], r.HomeTeam, r.AwayTeam)
            # liquidez: valor final de Gamma (proxy, no histórico). Filtro conservador, no exacto.
            liq = float(m.get("liquidityNum") or m.get("liquidity") or 0)
            rows.append({"ts": ts, "match_id": f"{r.date}_{r.HomeTeam}_{r.AwayTeam}", "outcome": o,
                         "question": mk["question"], "p_pm": p, "y": int(r.FTR == o), "liquidity": liq,
                         **{c: r[c] for c in ODDS}})
        log(f"[{k}] {r.HomeTeam}-{r.AwayTeam}: ok")
    return pd.DataFrame(rows)


def _odds(p, margin, rng):
    """Probabilidades -> cuotas con margen y sesgo favorito-longshot (para que el método de-vig importe)."""
    q = p ** 0.93
    q = q / q.sum() * (1 + margin)
    return 1 / q


def synthetic(n_matches=1500, pm_noise=0.35, open_noise=0.20, close_noise=0.10, margin=0.03,
              per_day=6, seed=1):
    """Simulación: el cierre de la casa es la fuente más precisa, la apertura menos y PM la que menos."""
    rng = np.random.default_rng(seed)
    rows = []
    lg = lambda x: np.log(x / (1 - x))
    sg = lambda x: 1 / (1 + np.exp(-x))
    for i in range(n_matches):
        true = rng.dirichlet([3, 2, 2.5])
        res = rng.choice(3, p=true)
        noisy = lambda s: (lambda v: v / v.sum())(sg(lg(true) + rng.normal(0, s, 3)))
        oc, oo = _odds(noisy(close_noise), margin, rng), _odds(noisy(open_noise), margin, rng)
        liq = float(rng.lognormal(8, 1.2))
        for j, o in enumerate("HDA"):
            rows.append({"ts": int(i * 86400 / per_day), "match_id": i, "outcome": o,
                         "p_pm": float(sg(lg(true[j]) + rng.normal(0, pm_noise))), "y": int(res == j),
                         "liquidity": liq, "oH": oc[0], "oD": oc[1], "oA": oc[2],
                         "ooH": oo[0], "ooD": oo[1], "ooA": oo[2]})
    return pd.DataFrame(rows)
