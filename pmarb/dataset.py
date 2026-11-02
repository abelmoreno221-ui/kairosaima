"""Dataset histórico: football-data (apertura + cierre de UNA casa) x Polymarket.
Fase 1 (barata): emparejar y medir cobertura -> aborta si es insuficiente.
Fase 2, dos puntos de decisión por mercado:
  - apertura: precio PM y volumen a T-24h (la apertura de football-data es de T-24h a T-72h).
  - cierre:   precio PM y volumen a T-10min (solo investigación).
Volumen = USDC negociados en los `vol_window_h` previos a cada punto (data-api, una vez por mercado).
Si /trades se agota antes de cubrir el tramo, el volumen queda NaN (desconocido), no se descarta."""
from datetime import datetime, time
import numpy as np
import pandas as pd
from . import polymarket as pm
from .matching import match_event, outcome_of

ODDS = ["oH", "oD", "oA", "ooH", "ooD", "ooA"]
DECISION_H = 24


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
    out = fd[fd._ev.notna()]
    out.attrs["coverage"] = rep.reset_index()
    return out


def build(fd: pd.DataFrame, tag_slug="soccer", close_lead_min=10, decision_h=DECISION_H,
          vol_window_h=168, kickoff_utc_hour=12, min_rate=0.30, min_matches=150, log=print):
    """Devuelve (dataset, cobertura_por_liga). decision_h: punto de decisión con la apertura (24 o 6)."""
    matched = coverage(fd, list(pm.iter_events(tag_slug, closed=True)), min_rate, min_matches, log)
    W = vol_window_h * 3600
    rows = []
    for k, r in matched.iterrows():
        for m in r._ev.get("markets", []):
            mk = pm.parse_market(m)
            if not mk["yes_token"]:
                continue
            ko = pd.Timestamp(mk["start"]) if mk["start"] else pd.Timestamp(datetime.combine(r.date, time(kickoff_utc_hour)))
            ko = int((ko.tz_convert("UTC") if ko.tzinfo else ko.tz_localize("UTC")).timestamp())
            t_open, t_close = ko - decision_h * 3600, ko - close_lead_min * 60
            p_open, p_close = pm.price_at(mk["yes_token"], t_open), pm.price_at(mk["yes_token"], t_close)
            if p_open is None and p_close is None:
                continue
            tc = pm.trades(mk["condition_id"], until_ts=t_open - W) if mk["condition_id"] else None
            vol = lambda t: (pm.volume_window(tc, t, W) if tc else None)
            o = outcome_of(mk["question"], r.HomeTeam, r.AwayTeam)
            rows.append({"ts_open": t_open, "ts_close": t_close,
                         "match_id": f"{r.league}_{r.date}_{r.HomeTeam}_{r.AwayTeam}", "league": r.league,
                         "source": r.source, "outcome": o, "question": mk["question"], "y": int(r.FTR == o),
                         "p_pm_open": p_open, "p_pm_close": p_close, "vol_open": vol(t_open),
                         "vol_close": vol(t_close), "decision_h": decision_h, **{c: r[c] for c in ODDS}})
        log(f"[{k}] {r.HomeTeam}-{r.AwayTeam}: ok")
    return pd.DataFrame(rows), matched.attrs["coverage"]


def _odds(p, margin):
    q = p ** 0.93
    return 1 / (q / q.sum() * (1 + margin))


def synthetic(n_matches=1500, pm_noise=0.35, open_noise=0.20, close_noise=0.10, margin=0.03,
              per_day=6, seed=1):
    """Simulación: cierre de la casa > apertura > PM en precisión. Fuente 'SIM'."""
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
            ts = int(i * 86400 / per_day)
            p_close = float(sg(lg(true[j]) + rng.normal(0, pm_noise)))
            rows.append({"ts_open": ts - 86400, "ts_close": ts, "match_id": i, "source": "SIM", "outcome": o,
                         "league": ("E0", "SP1", "I1", "D1", "F1")[i % 5], "decision_h": 24,
                         "p_pm_open": float(sg(lg(p_close) + rng.normal(0, 0.1))), "p_pm_close": p_close,
                         "y": int(res == j), "vol_open": vol * 0.6, "vol_close": vol,
                         "oH": oc[0], "oD": oc[1], "oA": oc[2], "ooH": oo[0], "ooD": oo[1], "ooA": oo[2]})
    return pd.DataFrame(rows)
