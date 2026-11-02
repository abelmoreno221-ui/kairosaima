"""Dataset histórico: football-data (apertura + cierre de una casa) x Polymarket.
Fase 1 (barata): emparejar y medir cobertura -> aborta si es insuficiente.
Fase 2: precio pre-partido (CLOB) y volumen negociado antes de T (data-api, cacheado por mercado)."""
from datetime import datetime, time
import numpy as np
import pandas as pd
from . import polymarket as pm
from .matching import match_event, outcome_of

ODDS = ["oH", "oD", "oA", "ooH", "ooD", "ooA"]


class LowCoverage(SystemExit):
    pass


def coverage(fd: pd.DataFrame, events: list, min_rate=0.30, min_matches=150, log=print):
    by_day = {}
    for e in events:
        by_day.setdefault((e.get("startDate") or e.get("endDate") or "")[:10], []).append(e)
    pairs = []
    for _, r in fd.iterrows():
        cands = sum((by_day.get(str(pd.Timestamp(r.date) + pd.Timedelta(days=o))[:10], []) for o in (-1, 0, 1)), [])
        pairs.append(match_event(r.HomeTeam, r.AwayTeam, cands))
    fd = fd.assign(_ev=pairs)
    rep = fd.groupby("league")._ev.agg(total="size", matched=lambda x: x.notna().sum())
    rep["rate"] = (rep.matched / rep.total).round(3)
    log(rep.to_string())
    n, rate = int(rep.matched.sum()), rep.matched.sum() / max(rep.total.sum(), 1)
    if n < min_matches or rate < min_rate:
        raise LowCoverage(f"ABORTADO: solo {n} partidos emparejados ({rate:.1%}) "
                          f"< mínimo {min_matches} partidos / {min_rate:.0%}. No se calculan cifras.")
    return fd[fd._ev.notna()]


def build(fd: pd.DataFrame, tag_slug="soccer", lead_min=10, kickoff_utc_hour=12,
          min_rate=0.30, min_matches=150, log=print):
    matched = coverage(fd, list(pm.iter_events(tag_slug, closed=True)), min_rate, min_matches, log)
    rows = []
    for k, r in matched.iterrows():
        for m in r._ev.get("markets", []):
            mk = pm.parse_market(m)
            if not mk["yes_token"]:
                continue
            ko = pd.Timestamp(mk["start"]) if mk["start"] else pd.Timestamp(datetime.combine(r.date, time(kickoff_utc_hour)))
            ko = ko.tz_convert("UTC") if ko.tzinfo else ko.tz_localize("UTC")
            ts = int(ko.timestamp()) - lead_min * 60
            p = pm.price_at(mk["yes_token"], ts)
            if p is None or not 0 < p < 1:
                continue
            vol = pm.volume_before(pm.trades(mk["condition_id"]), ts) if mk["condition_id"] else 0.0
            o = outcome_of(mk["question"], r.HomeTeam, r.AwayTeam)
            rows.append({"ts": ts, "match_id": f"{r.league}_{r.date}_{r.HomeTeam}_{r.AwayTeam}",
                         "league": r.league, "outcome": o, "question": mk["question"], "p_pm": p,
                         "y": int(r.FTR == o), "vol_before": vol, **{c: r[c] for c in ODDS}})
        log(f"[{k}] {r.HomeTeam}-{r.AwayTeam}: ok")
    return pd.DataFrame(rows)


def _odds(p, margin):
    q = p ** 0.93
    return 1 / (q / q.sum() * (1 + margin))


def synthetic(n_matches=1500, pm_noise=0.35, open_noise=0.20, close_noise=0.10, margin=0.03,
              per_day=6, seed=1):
    """Simulación: cierre de la casa > apertura > PM en precisión."""
    rng = np.random.default_rng(seed)
    lg = lambda x: np.log(x / (1 - x))
    sg = lambda x: 1 / (1 + np.exp(-x))
    rows = []
    for i in range(n_matches):
        true = rng.dirichlet([3, 2, 2.5])
        res = rng.choice(3, p=true)
        noisy = lambda s: (lambda v: v / v.sum())(sg(lg(true) + rng.normal(0, s, 3)))
        oc, oo = _odds(noisy(close_noise), margin), _odds(noisy(open_noise), margin)
        vol = float(rng.lognormal(7.5, 1.5))
        for j, o in enumerate("HDA"):
            rows.append({"ts": int(i * 86400 / per_day), "match_id": i, "outcome": o,
                         "p_pm": float(sg(lg(true[j]) + rng.normal(0, pm_noise))), "y": int(res == j),
                         "vol_before": vol, "oH": oc[0], "oD": oc[1], "oA": oc[2],
                         "ooH": oo[0], "ooD": oo[1], "ooA": oo[2]})
    return pd.DataFrame(rows)
