"""Dataset histórico: football-data (apertura + cierre de UNA casa) x Polymarket.
Fase 1 (barata): emparejar y medir cobertura -> aborta si es insuficiente.
Fase 2, tres puntos de decisión por mercado (precio PM y volumen en cada uno):
  - 24:    T-24h  -> con la apertura de football-data (T-24h..T-72h): validación conservadora.
  - 6:     T-6h   -> horizonte operativo (el del escáner).
  - close: T-10min -> con el cierre: techo de investigación.
Volumen = USDC negociados en los `vol_window_h` previos a cada punto (data-api, una vez por mercado).
Si /trades se agota antes de cubrir el tramo, el volumen queda NaN (desconocido), no se descarta."""
from datetime import datetime, time
import numpy as np
import pandas as pd
from . import polymarket as pm
from .matching import match_event, outcome_of

ODDS = ["oH", "oD", "oA", "ooH", "ooD", "ooA"]
POINTS = {"24": 24 * 3600, "6": 6 * 3600, "close": 10 * 60}  # segundos antes del saque


def match_day(e: dict) -> str:
    """Fecha del PARTIDO. Ojo: en Gamma 'startDate' es cuándo se creó el mercado (~2 semanas antes)."""
    return (e.get("eventDate") or e.get("startTime") or e.get("endDate") or "")[:10]


class LowCoverage(SystemExit):
    pass


def coverage(fd: pd.DataFrame, events: list, min_rate=0.30, min_matches=150, log=print):
    by_day = {}
    for e in events:
        by_day.setdefault(match_day(e), []).append(e)
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


def build(fd: pd.DataFrame, tag_slug="soccer", vol_window_h=168, kickoff_utc_hour=12,
          min_rate=0.30, min_matches=150, log=print):
    """Devuelve (dataset, cobertura_por_liga). Columnas ts_/p_pm_/vol_ para cada punto de POINTS."""
    lo = (pd.Timestamp(fd.date.min()) - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    hi = (pd.Timestamp(fd.date.max()) + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    evs = list(pm.iter_events(tag_slug, closed=True, start=lo, end=hi))
    log(f"Eventos Polymarket '{tag_slug}' cerrados entre {lo} y {hi}: {len(evs)}")
    matched = coverage(fd, evs, min_rate, min_matches, log)
    W = vol_window_h * 3600
    rows = []
    for k, r in matched.iterrows():
        for m in r._ev.get("markets", []):
            mk = pm.parse_market(m)
            if not mk["yes_token"]:
                continue
            ko = pd.Timestamp(mk["start"]) if mk["start"] else pd.Timestamp(datetime.combine(r.date, time(kickoff_utc_hour)))
            ko = int((ko.tz_convert("UTC") if ko.tzinfo else ko.tz_localize("UTC")).timestamp())
            ts = {k_: ko - off for k_, off in POINTS.items()}
            prices = dict(zip(ts, pm.prices_at(mk["yes_token"], list(ts.values()))))
            if all(p is None for p in prices.values()):
                continue
            tc = pm.trades(mk["condition_id"], until_ts=min(ts.values()) - W) if mk["condition_id"] else None
            o = outcome_of(mk["question"], r.HomeTeam, r.AwayTeam)
            row = {"match_id": f"{r.league}_{r.date}_{r.HomeTeam}_{r.AwayTeam}", "league": r.league,
                   "source": r.source, "outcome": o, "question": mk["question"], "y": int(r.FTR == o),
                   **{c: r[c] for c in ODDS}}
            for k_, t in ts.items():
                row[f"ts_{k_}"], row[f"p_pm_{k_}"] = t, prices[k_]
                row[f"vol_{k_}"] = pm.volume_window(tc, t, W) if tc else None
            rows.append(row)
        log(f"[{k}] {r.HomeTeam}-{r.AwayTeam}: ok")
    return validate(pd.DataFrame(rows), log), matched.attrs["coverage"]


def validate(df: pd.DataFrame, log=print) -> pd.DataFrame:
    """Quita partidos sin exactamente un mercado H, uno D y uno A (o con resultado != 1)."""
    if df.empty:
        return df
    ok_set = df.groupby("match_id").outcome.apply(lambda x: sorted(map(str, x)) == ["A", "D", "H"])
    ok_y = df.groupby("match_id").y.sum() == 1
    good = ok_set & ok_y
    bad = good[~good].index.tolist()
    if bad:
        log(f"VALIDACIÓN: {len(bad)} partidos excluidos por mapeo H/D/A inconsistente: {bad}")
    return df[df.match_id.isin(good[good].index)].reset_index(drop=True)


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
            p6 = float(sg(lg(p_close) + rng.normal(0, 0.05)))
            rows.append({"ts_24": ts - 86400, "ts_6": ts - 6 * 3600, "ts_close": ts, "match_id": i,
                         "source": "SIM", "outcome": o, "league": ("E0", "SP1", "I1", "D1", "F1")[i % 5],
                         "p_pm_24": float(sg(lg(p6) + rng.normal(0, 0.1))), "p_pm_6": p6, "p_pm_close": p_close,
                         "y": int(res == j), "vol_24": vol * 0.6, "vol_6": vol * 0.8, "vol_close": vol,
                         "oH": oc[0], "oD": oc[1], "oA": oc[2], "ooH": oo[0], "ooD": oo[1], "ooA": oo[2]})
    return pd.DataFrame(rows)
