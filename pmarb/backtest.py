"""Backtest walk-forward (sin lookahead) y rejilla línea x método de de-vig.
Dataset: ts, match_id, outcome (H/D/A), p_pm, y, oH/oD/oA (cierre), ooH/ooD/ooA (apertura), liquidity.
"""
from dataclasses import dataclass, asdict, replace
import numpy as np
import pandas as pd
from .devig import METHODS
from .model import Blend, report

LINES = {"close": ("oH", "oD", "oA"), "open": ("ooH", "ooD", "ooA")}


@dataclass
class Cfg:
    min_train: int = 300          # filas mínimas antes de operar
    step: int = 100               # re-entrenar cada N filas
    half_spread: float = 0.01     # ask = mid + half_spread
    fee_rate: float = 0.05        # taker: fee = shares * 0.05 * c * (1-c)  (2,5 % del stake a 0,50)
    min_edge: float = 0.04        # EV neto mínimo por $ apostado (tras fee y spread)
    min_liquidity: float = 1000.0  # filtro de liquidez del mercado (USDC)
    kelly: float = 0.25
    max_stake: float = 0.02
    max_stake_abs: float = 200.0
    bankroll: float = 1000.0


def fee_per_share(c, rate):
    """Comisión de taker de Polymarket por acción comprada a precio c."""
    return rate * c * (1 - c)


def decide(p, mid, cfg: Cfg):
    """(lado, precio, ev_neto_por_$, prob_lado) o None. Compra YES o NO como taker."""
    best = None
    for side, q, c in (("YES", p, mid + cfg.half_spread), ("NO", 1 - p, 1 - mid + cfg.half_spread)):
        c = min(c, 0.99)
        ev = (q - c - fee_per_share(c, cfg.fee_rate)) / c
        if best is None or ev > best[2]:
            best = (side, c, ev, q)
    return best if best[2] > cfg.min_edge else None


def with_book(df: pd.DataFrame, line="close", method="shin") -> pd.DataFrame:
    """Añade p_book quitando el margen con la línea y el método indicados."""
    cols = LINES[line]
    df = df.dropna(subset=list(cols)).copy()
    f, idx = METHODS[method], {"H": 0, "D": 1, "A": 2}
    df["p_book"] = [f([r[cols[0]], r[cols[1]], r[cols[2]]])[idx[r.outcome]] for _, r in df.iterrows()]
    return df


def weekly(trades: pd.DataFrame, ts_from, ts_to):
    weeks = max(1, int(np.ceil((ts_to - ts_from) / (7 * 86400))))
    counts = np.zeros(weeks, int)
    if len(trades):
        np.add.at(counts, ((trades.ts.values - ts_from) // (7 * 86400)).astype(int).clip(0, weeks - 1), 1)
    return {"weeks": weeks, "mean": float(counts.mean()), "median": float(np.median(counts)),
            "p10": float(np.percentile(counts, 10)), "zero_weeks": int((counts == 0).sum())}


def run(df: pd.DataFrame, cfg: Cfg = Cfg()):
    df = df.sort_values("ts").reset_index(drop=True)
    if "liquidity" not in df:
        df["liquidity"] = np.inf
    bank, trades, blocks = cfg.bankroll, [], []
    preds = np.full(len(df), np.nan)
    for start in range(cfg.min_train, len(df), cfg.step):
        te = df.iloc[start:start + cfg.step]
        tr = df.iloc[:start]
        tr = tr[~tr.match_id.isin(te.match_id)]
        m = Blend().fit(tr.p_pm.values, tr.p_book.values, tr.y.values)
        blocks.append({"block": len(blocks), "n_train": len(tr), "n_test": len(te),
                       "w": m.w.round(3).tolist()})
        preds[te.index] = m.predict(te.p_pm.values, te.p_book.values)
        for i, r in te.iterrows():
            if r.liquidity < cfg.min_liquidity:
                continue
            d = decide(preds[i], r.p_pm, cfg)
            if not d:
                continue
            side, c, ev, q = d
            f = min(cfg.kelly * (q - c) / (1 - c), cfg.max_stake)
            stake = min(bank * f, cfg.max_stake_abs)
            shares = stake / c
            fee = shares * fee_per_share(c, cfg.fee_rate)
            win = r.y == (1 if side == "YES" else 0)
            pnl = shares * win - stake - fee
            bank += pnl
            trades.append({"ts": r.ts, "match_id": r.match_id, "side": side, "price": c, "p_model": q,
                           "ev": ev, "stake": stake, "fee": fee, "pnl": pnl, "bank": bank})
    t = pd.DataFrame(trades)
    mask = ~np.isnan(preds)
    oos = df[mask]
    s = {
        "oos_logloss": {"pm": report(oos.p_pm, oos.y)["logloss"], "book": report(oos.p_book, oos.y)["logloss"],
                        "model": report(preds[mask], oos.y)["logloss"]},
        "signals": len(t), "signals_per_week": weekly(t, oos.ts.min(), oos.ts.max()),
        "roi": float(t.pnl.sum() / t.stake.sum()) if len(t) else 0.0,
        "fees_paid": float(t.fee.sum()) if len(t) else 0.0,
        "final_bank": round(bank, 2), "blocks": blocks, "cfg": asdict(cfg),
    }
    ll = s["oos_logloss"]
    s["beats_pm"] = ll["model"] < ll["pm"]        # imprescindible: si no, no hay edge contra PM
    s["beats_book"] = ll["model"] < ll["book"]    # informativo: ¿la mezcla aporta sobre la casa sola?
    if len(t) > 30:
        rng = np.random.default_rng(0)
        r = (t.pnl / t.stake).values
        boot = [rng.choice(r, len(r)).mean() for _ in range(2000)]
        s["roi_ci95"] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    return s, t


def grid(df: pd.DataFrame, cfg: Cfg = Cfg(), methods=("shin", "prop", "log")):
    """Rejilla línea {close, open} x método de-vig, con veredictos."""
    res = {}
    for line in LINES:
        if df[list(LINES[line])].isna().all().all():
            continue
        for meth in methods:
            res[(line, meth)], _ = run(with_book(df, line, meth), cfg)
    v = {}
    for line in {l for l, _ in res}:
        rois = [res[(line, m)]["roi"] for m in methods if (line, m) in res]
        v[f"{line}_roi_spread_pts"] = round(100 * (max(rois) - min(rois)), 2)
        v[f"{line}_method_sensitive"] = v[f"{line}_roi_spread_pts"] > 1.0
    sig = lambda s: s.get("roi_ci95", [0, 0])[0] > 0 and s["beats_pm"]
    close_ok = any(sig(s) for (l, _), s in res.items() if l == "close")
    open_ok = any(sig(s) for (l, _), s in res.items() if l == "open")
    v["verdict"] = ("estrategia candidata (edge en apertura)" if open_ok else
                    "solo investigación (edge solo en cierre)" if close_ok else "sin edge")
    return res, v


def table(res) -> pd.DataFrame:
    return pd.DataFrame([{
        "line": l, "devig": m, "signals": s["signals"],
        "sig/wk_mean": round(s["signals_per_week"]["mean"], 2), "sig/wk_med": s["signals_per_week"]["median"],
        "zero_wks": f'{s["signals_per_week"]["zero_weeks"]}/{s["signals_per_week"]["weeks"]}',
        "roi%": round(100 * s["roi"], 2),
        "ci95%": [round(100 * x, 1) for x in s["roi_ci95"]] if "roi_ci95" in s else None,
        "ll_model": round(s["oos_logloss"]["model"], 4), "ll_pm": round(s["oos_logloss"]["pm"], 4),
        "ll_book": round(s["oos_logloss"]["book"], 4), "beats_pm": s["beats_pm"], "beats_book": s["beats_book"],
        "blocks": len(s["blocks"]),
        "n_train_first/last": f'{s["blocks"][0]["n_train"]}/{s["blocks"][-1]["n_train"]}' if s["blocks"] else "-",
    } for (l, m), s in res.items()])
