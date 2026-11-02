"""Escáner en vivo (PAPER TRADING): casa (The Odds API) vs order book de Polymarket.
Punto de decisión por defecto en T-6h (horizonte operativo; el backtest lo evalúa con --decision-h 6):
solo evalúa partidos cuyo saque está a decision_h ± tol_h horas. Lánzalo cada hora.
No envía órdenes. Registra señales en CSV para validarlas a posteriori (forward test)."""
import json
import os
from datetime import datetime, timezone
import pandas as pd
from . import polymarket as pm
from .backtest import Cfg, decide
from .devig import METHODS
from .matching import match_event, outcome_of
from .model import Blend


def scan(sport="soccer_epl", tag_slug="soccer", weights_path="weights.json", out="signals.csv",
         cfg: Cfg = Cfg(), devig="shin", decision_h=6.0, tol_h=1.0):
    model = Blend()
    model.w = json.load(open(weights_path))["w"] if os.path.exists(weights_path) else [0.0, 0.0, 1.0]
    from .odds import odds_api_live
    book = odds_api_live(sport)
    evs = list(pm.iter_events(tag_slug, closed=False, max_pages=10))
    now = datetime.now(timezone.utc).isoformat()
    sigs = []
    t_now = pd.Timestamp.now(tz="UTC")
    for _, r in book.dropna().iterrows():
        htk = (pd.Timestamp(r.start) - t_now).total_seconds() / 3600
        if abs(htk - decision_h) > tol_h:
            continue
        ev = match_event(r.home, r.away, evs)
        if not ev:
            continue
        probs = dict(zip("HDA", METHODS[devig]([r.oH, r.oD, r.oA])))
        for m in ev.get("markets", []):
            mk = pm.parse_market(m)
            if not mk["yes_token"]:
                continue
            tob = pm.top_of_book(mk["yes_token"])
            if not tob["bid"] or not tob["ask"]:
                continue
            mid = (tob["bid"][0] + tob["ask"][0]) / 2
            o = outcome_of(mk["question"], r.home, r.away)
            p = float(model.predict([mid], [probs[o]])[0])
            c = Cfg(**{**cfg.__dict__, "half_spread": (tob["ask"][0] - tob["bid"][0]) / 2})
            d = decide(p, mid, c, htk)
            if d:
                side, price, evv, q = d
                sigs.append({"time": now, "match": f"{r.home} vs {r.away}", "kickoff": r.start,
                             "hours_to_kickoff": round(htk, 2), "outcome": o,
                             "side": side, "price": round(price, 3), "p_model": round(q, 3),
                             "p_book": round(probs[o], 3), "ev": round(evv, 4), "token": mk["yes_token"]})
    df = pd.DataFrame(sigs)
    if len(df):
        df.to_csv(out, mode="a", header=not os.path.exists(out), index=False)
    return df
