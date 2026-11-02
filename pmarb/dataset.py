"""Construye el dataset histórico: football-data (Pinnacle cierre) x Polymarket (precio pre-partido)."""
from datetime import datetime, time, timezone
import pandas as pd
from . import polymarket as pm
from .devig import METHODS
from .matching import match_event, outcome_of


def build(fd: pd.DataFrame, tag_slug="soccer", lead_min=10, devig="shin", kickoff_utc_hour=None, log=print):
    evs = list(pm.iter_events(tag_slug, closed=True))
    by_day = {}
    for e in evs:
        d = (e.get("startDate") or e.get("endDate") or "")[:10]
        by_day.setdefault(d, []).append(e)
    rows = []
    for k, r in fd.iterrows():
        cands = sum((by_day.get(str(pd.Timestamp(r.date) + pd.Timedelta(days=o))[:10], []) for o in (-1, 0, 1)), [])
        ev = match_event(r.HomeTeam, r.AwayTeam, cands)
        if not ev:
            continue
        probs = dict(zip("HDA", METHODS[devig]([r.oH, r.oD, r.oA])))
        for m in ev.get("markets", []):
            mk = pm.parse_market(m)
            if not mk["yes_token"]:
                continue
            start = mk["start"]
            if start:
                ko = pd.Timestamp(start).tz_convert("UTC") if pd.Timestamp(start).tzinfo else pd.Timestamp(start, tz="UTC")
            else:
                ko = pd.Timestamp(datetime.combine(r.date, time(kickoff_utc_hour or 12)), tz="UTC")
            ts = int(ko.timestamp()) - lead_min * 60
            p = pm.price_at(mk["yes_token"], ts)
            if p is None or not 0 < p < 1:
                continue
            o = outcome_of(mk["question"], r.HomeTeam, r.AwayTeam)
            rows.append({"ts": ts, "match_id": f"{r.date}_{r.HomeTeam}_{r.AwayTeam}", "outcome": o,
                         "question": mk["question"], "p_pm": p, "p_book": probs[o], "y": int(r.FTR == o)})
        log(f"[{k}] {r.HomeTeam}-{r.AwayTeam}: ok")
    return pd.DataFrame(rows)


def synthetic(n_matches=1500, pm_noise=0.35, book_noise=0.12, seed=1):
    """Datos simulados para probar el pipeline: la casa sharp es más precisa que PM."""
    import numpy as np
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_matches):
        true = rng.dirichlet([3, 2, 2.5])
        res = rng.choice(3, p=true)
        for j, o in enumerate("HDA"):
            lt = np.log(true[j] / (1 - true[j]))
            rows.append({"ts": i * 3600, "match_id": i, "outcome": o,
                         "p_pm": 1 / (1 + np.exp(-(lt + rng.normal(0, pm_noise)))),
                         "p_book": 1 / (1 + np.exp(-(lt + rng.normal(0, book_noise)))), "y": int(res == j)})
    return pd.DataFrame(rows)
