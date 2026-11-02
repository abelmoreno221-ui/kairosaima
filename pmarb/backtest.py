"""Backtest walk-forward (sin lookahead): entrena con el pasado, opera en el siguiente bloque.
Dataset: columnas ts, match_id, p_pm (precio YES en PM), p_book (prob. justa casa), y (0/1).
"""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from .model import Blend, report


@dataclass
class Cfg:
    min_train: int = 300        # filas mínimas antes de operar
    step: int = 100             # re-entrenar cada N filas
    half_spread: float = 0.01   # ask = p_pm + half_spread (conservador si solo tienes mid)
    fee_rate: float = 0.0       # fee Polymarket: fee_rate * min(c, 1-c) por share (revisa docs actuales)
    min_edge: float = 0.02      # EV mínimo por share después de costes
    kelly: float = 0.25         # Kelly fraccional
    max_stake: float = 0.02     # % máximo del bankroll por apuesta
    bankroll: float = 1000.0
    max_stake_abs: float = 200.0  # tope en USDC por apuesta (liquidez real del libro)


def fee(c, rate):
    return rate * np.minimum(c, 1 - c)


def decide(p, p_pm, cfg: Cfg):
    """Devuelve (lado, precio, ev_por_share) o None. Compra YES o NO en Polymarket."""
    ask_yes = min(p_pm + cfg.half_spread, 0.99)
    ask_no = min(1 - p_pm + cfg.half_spread, 0.99)
    ev_yes = p - ask_yes - fee(ask_yes, cfg.fee_rate)
    ev_no = (1 - p) - ask_no - fee(ask_no, cfg.fee_rate)
    side, c, ev, q = ("YES", ask_yes, ev_yes, p) if ev_yes >= ev_no else ("NO", ask_no, ev_no, 1 - p)
    return (side, c, ev, q) if ev > cfg.min_edge else None


def run(df: pd.DataFrame, cfg: Cfg = Cfg()):
    df = df.sort_values("ts").reset_index(drop=True)
    bank, trades, preds = cfg.bankroll, [], np.full(len(df), np.nan)
    for start in range(cfg.min_train, len(df), cfg.step):
        tr, te = df.iloc[:start], df.iloc[start:start + cfg.step]
        # no entrenar con partidos que aún no han terminado al inicio del bloque
        tr = tr[~tr.match_id.isin(te.match_id)]
        m = Blend().fit(tr.p_pm.values, tr.p_book.values, tr.y.values)
        preds[te.index] = m.predict(te.p_pm.values, te.p_book.values)
        for i, r in te.iterrows():
            d = decide(preds[i], r.p_pm, cfg)
            if not d:
                continue
            side, c, ev, q = d
            f = min(cfg.kelly * (q - c) / (1 - c), cfg.max_stake)
            stake = min(bank * f, cfg.max_stake_abs)
            shares = stake / c
            win = r.y == (1 if side == "YES" else 0)
            pnl = shares * (1.0 if win else 0.0) - stake - shares * fee(c, cfg.fee_rate)
            bank += pnl
            trades.append({"ts": r.ts, "match_id": r.match_id, "side": side, "price": c, "p_model": q,
                           "ev": ev, "stake": stake, "pnl": pnl, "bank": bank, "w": m.w.round(3).tolist()})
    t = pd.DataFrame(trades)
    mask = ~np.isnan(preds)
    oos = df[mask]
    summary = {
        "oos_pm": report(oos.p_pm, oos.y), "oos_book": report(oos.p_book, oos.y),
        "oos_model": report(preds[mask], oos.y),
        "trades": len(t), "final_bank": round(bank, 2),
        "roi_on_turnover": float(t.pnl.sum() / t.stake.sum()) if len(t) else 0.0,
        "max_drawdown": float((t.bank.cummax() - t.bank).max() / t.bank.cummax().max()) if len(t) else 0.0,
        "hit_rate": float((t.pnl > 0).mean()) if len(t) else 0.0, "cfg": asdict(cfg),
    }
    if len(t) > 30:  # bootstrap: ¿el ROI es distinto de 0 o es suerte?
        rng = np.random.default_rng(0)
        r = (t.pnl / t.stake).values
        boot = [rng.choice(r, len(r)).mean() for _ in range(2000)]
        summary["roi_per_trade_ci95"] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    return summary, t
