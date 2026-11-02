"""Backtest walk-forward (sin lookahead): rejilla línea {close, open} x de-vig {shin, prop, log}
x medio spread {1, 2, 3, 5 céntimos [, calibrado]}.

Dataset: ts, match_id, outcome (H/D/A), p_pm (precio YES ~ mid), y, oH/oD/oA (cierre),
ooH/ooD/ooA (apertura), vol_before (USDC negociados en el mercado ANTES de ts).

Convenciones de costes (verificadas en tests):
- half_spread = MEDIO spread: comprar como taker cuesta ask = mid + half_spread, se paga al entrar.
- Comisión de taker: shares * fee_rate * c * (1-c), cobrada en la entrada al hacer match,
  gane o pierda la apuesta.
- El umbral de EV se evalúa sobre el ask (precio de compra), nunca sobre el mid.
"""
from dataclasses import dataclass, asdict, replace
import json
import numpy as np
import pandas as pd
from .devig import METHODS
from .model import Blend, brier, report

LINES = {"close": ("oH", "oD", "oA"), "open": ("ooH", "ooD", "ooA")}
SPREADS = (0.01, 0.02, 0.03, 0.05)


@dataclass
class Cfg:
    min_train: int = 300
    step: int = 100
    half_spread: float = 0.01      # MEDIO spread en $ por acción (no el spread completo)
    spread_table: tuple | None = None  # ((lo, hi, half), ...) por tramo de precio, de `spreads summary`
    fee_rate: float = 0.05
    min_edge: float = 0.04         # EV neto mínimo por $ apostado, sobre el ask, tras fee
    min_volume: float = 1000.0     # USDC negociados en el mercado antes de la señal
    kelly: float = 0.25
    max_stake: float = 0.02
    max_stake_abs: float = 200.0
    bankroll: float = 1000.0


def fee_per_share(c, rate):
    """Comisión de taker por acción comprada a precio c (se paga en la entrada)."""
    return rate * c * (1 - c)


def half_spread_at(mid, cfg: Cfg):
    if cfg.spread_table:
        for lo, hi, h in cfg.spread_table:
            if lo <= mid < hi:
                return h
    return cfg.half_spread


def decide(p, mid, cfg: Cfg):
    """(lado, ask, ev_neto_por_$, prob_lado) o None. Compra YES o NO como taker al ask."""
    best = None
    for side, q, m in (("YES", p, mid), ("NO", 1 - p, 1 - mid)):
        ask = min(m + half_spread_at(m, cfg), 0.99)
        ev = (q - ask - fee_per_share(ask, cfg.fee_rate)) / ask   # sobre el precio de compra
        if best is None or ev > best[2]:
            best = (side, ask, ev, q)
    return best if best[2] > cfg.min_edge else None


def settle(stake, ask, won, rate):
    """PnL de una compra de `stake` $ al ask. La comisión se descuenta siempre."""
    shares = stake / ask
    fee = shares * fee_per_share(ask, rate)
    return shares * won - stake - fee, fee


def with_book(df: pd.DataFrame, line="close", method="shin") -> pd.DataFrame:
    cols = LINES[line]
    df = df.dropna(subset=list(cols)).copy()
    f, idx = METHODS[method], {"H": 0, "D": 1, "A": 2}
    df["p_book"] = [f([r[cols[0]], r[cols[1]], r[cols[2]]])[idx[r.outcome]] for _, r in df.iterrows()]
    return df.sort_values("ts").reset_index(drop=True)


def predict_oos(df: pd.DataFrame, cfg: Cfg):
    """Predicciones fuera de muestra y tamaño de entrenamiento de cada bloque."""
    preds, blocks = np.full(len(df), np.nan), []
    for start in range(cfg.min_train, len(df), cfg.step):
        te = df.iloc[start:start + cfg.step]
        tr = df.iloc[:start]
        tr = tr[~tr.match_id.isin(te.match_id)]
        m = Blend().fit(tr.p_pm.values, tr.p_book.values, tr.y.values)
        blocks.append({"block": len(blocks), "n_train": len(tr), "n_test": len(te), "w": m.w.round(3).tolist()})
        preds[te.index] = m.predict(te.p_pm.values, te.p_book.values)
    return preds, blocks


def simulate(df: pd.DataFrame, preds, cfg: Cfg) -> pd.DataFrame:
    bank, rows = cfg.bankroll, []
    vol = df["vol_before"].values if "vol_before" in df else np.full(len(df), np.inf)
    for i, r in enumerate(df.itertuples()):
        if np.isnan(preds[i]) or vol[i] < cfg.min_volume:
            continue
        d = decide(preds[i], r.p_pm, cfg)
        if not d:
            continue
        side, ask, ev, q = d
        eff = ask + fee_per_share(ask, cfg.fee_rate)          # precio efectivo con comisión
        f = min(max(cfg.kelly * (q - eff) / (1 - eff), 0), cfg.max_stake)
        stake = min(bank * f, cfg.max_stake_abs)
        won = int(r.y == (1 if side == "YES" else 0))
        pnl, fee = settle(stake, ask, won, cfg.fee_rate)
        bank += pnl
        bd = decide(r.p_book, r.p_pm, cfg)
        q_book = r.p_book if side == "YES" else 1 - r.p_book
        rows.append({"ts": r.ts, "match_id": r.match_id, "side": side, "ask": ask, "p_model": q,
                     "p_book": q_book, "won": won, "ev": ev, "stake": stake, "fee": fee, "pnl": pnl,
                     "bank": bank,
                     # con_pinnacle: la casa sola también dispara la misma operación.
                     # solo_modelo: la señal existe solo porque el blend se separa también de la casa.
                     "group": "con_pinnacle" if bd and bd[0] == side else "solo_modelo"})
    return pd.DataFrame(rows)


def _boot_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    b = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def weekly(t: pd.DataFrame, ts_from, ts_to):
    weeks = max(1, int(np.ceil((ts_to - ts_from + 1) / (7 * 86400))))
    c = np.zeros(weeks, int)
    if len(t):
        np.add.at(c, ((t.ts.values - ts_from) // (7 * 86400)).astype(int).clip(0, weeks - 1), 1)
    return {"weeks": weeks, "mean": float(c.mean()), "median": float(np.median(c)),
            "p10": float(np.percentile(c, 10)), "zero_weeks": int((c == 0).sum())}


def groups(t: pd.DataFrame):
    out = {}
    for g in ("solo_modelo", "con_pinnacle"):
        x = t[t.group == g] if len(t) else t
        out[g] = {"n": len(x),
                  "roi": float(x.pnl.sum() / x.stake.sum()) if len(x) else 0.0,
                  "roi_ci95": _boot_ci((x.pnl / x.stake).values) if len(x) > 30 else None,
                  "brier_model": brier(x.p_model, x.won) if len(x) else None,
                  "brier_book": brier(x.p_book, x.won) if len(x) else None}
    return out


def guardrail(df, preds):
    """Brier(modelo) - Brier(casa), bootstrap por partido. Positivo = el modelo es peor que la casa."""
    m = ~np.isnan(preds)
    d = pd.DataFrame({"mid": df.match_id[m], "d": (preds[m] - df.y[m]) ** 2 - (df.p_book[m] - df.y[m]) ** 2})
    per_match = d.groupby("mid").d.mean().values
    ci = _boot_ci(per_match)
    return {"brier_diff": float(per_match.mean()), "ci95": ci, "model_worse_than_book": ci[0] > 0}


def summarize(df, preds, t, blocks, cfg):
    m = ~np.isnan(preds)
    oos = df[m]
    ll = {"pm": report(oos.p_pm, oos.y)["logloss"], "book": report(oos.p_book, oos.y)["logloss"],
          "model": report(preds[m], oos.y)["logloss"]}
    s = {"oos_logloss": ll, "beats_pm": ll["model"] < ll["pm"], "beats_book": ll["model"] < ll["book"],
         "signals": len(t), "signals_per_week": weekly(t, oos.ts.min(), oos.ts.max()),
         "roi": float(t.pnl.sum() / t.stake.sum()) if len(t) else 0.0,
         "roi_ci95": _boot_ci((t.pnl / t.stake).values) if len(t) > 30 else None,
         "fees_paid": float(t.fee.sum()) if len(t) else 0.0, "final_bank": round(float(cfg.bankroll + (t.pnl.sum() if len(t) else 0)), 2),
         "groups": groups(t), "blocks": blocks, "cfg": asdict(cfg)}
    return s


def run(df: pd.DataFrame, cfg: Cfg = Cfg()):
    df = df.sort_values("ts").reset_index(drop=True)
    preds, blocks = predict_oos(df, cfg)
    t = simulate(df, preds, cfg)
    return summarize(df, preds, t, blocks, cfg), t


def _alive(s):
    return bool(s["roi_ci95"] and s["roi_ci95"][0] > 0 and s["beats_pm"])


def grid(df: pd.DataFrame, cfg: Cfg = Cfg(), methods=("shin", "prop", "log"), spreads=SPREADS):
    """res[(line, método, nivel_spread)] -> resumen. Nivel 'cal' = tabla calibrada si se pasa."""
    levels = [(f"{h:.2f}", replace(cfg, half_spread=h, spread_table=None)) for h in spreads]
    if cfg.spread_table:
        levels.append(("cal", cfg))
    res, guard = {}, {}
    for line in LINES:
        if df[list(LINES[line])].isna().all().all():
            continue
        for meth in methods:
            d = with_book(df, line, meth)
            preds, blocks = predict_oos(d, cfg)
            guard[(line, meth)] = guardrail(d, preds)
            for name, c in levels:
                res[(line, meth, name)] = summarize(d, preds, simulate(d, preds, c), blocks, c)
    v = {"edge_dies_at_half_spread": {}, "method_sensitive": {}, "guardrail": {}}
    for (line, meth), g in guard.items():
        lv = [n for n, _ in levels if (line, meth, n) in res]
        dead = next((n for n in lv if n != "cal" and not _alive(res[(line, meth, n)])), None)
        v["edge_dies_at_half_spread"][f"{line}/{meth}"] = dead or f">{lv[-1]}"
        v["guardrail"][f"{line}/{meth}"] = g
    for line in {k[0] for k in res}:
        for n, _ in levels:
            rois = [res[(line, m, n)]["roi"] for m in methods if (line, m, n) in res]
            v["method_sensitive"][f"{line}@{n}"] = round(100 * (max(rois) - min(rois)), 2)
    v["method_sensitive_any"] = any(x > 1.0 for x in v["method_sensitive"].values())
    v["guardrail_warning"] = [k for k, g in v["guardrail"].items() if g["model_worse_than_book"]]
    open_ok = any(_alive(s) for (l, _, n), s in res.items() if l == "open" and n in ("0.01", "cal"))
    close_ok = any(_alive(s) for (l, _, n), s in res.items() if l == "close" and n in ("0.01", "cal"))
    noise = any(s["groups"]["solo_modelo"]["n"] > 30 and not (s["groups"]["solo_modelo"]["roi_ci95"] or [0])[0] > 0
                for (l, _, n), s in res.items() if l == "open")
    v["solo_modelo_fails_open"] = noise
    v["verdict"] = ("estrategia candidata (edge en apertura)" if open_ok else
                    "solo investigación (edge solo en cierre)" if close_ok else "sin edge")
    return res, v


def table(res) -> pd.DataFrame:
    rows = []
    for (l, m, n), s in res.items():
        g = s["groups"]
        rows.append({
            "line": l, "devig": m, "half_spr": n, "signals": s["signals"],
            "sig/wk": round(s["signals_per_week"]["mean"], 2), "sig/wk_med": s["signals_per_week"]["median"],
            "zero_wks": f'{s["signals_per_week"]["zero_weeks"]}/{s["signals_per_week"]["weeks"]}',
            "roi%": round(100 * s["roi"], 2),
            "ci95%": [round(100 * x, 1) for x in s["roi_ci95"]] if s["roi_ci95"] else None,
            "solo_mod n/roi%": f'{g["solo_modelo"]["n"]}/{100 * g["solo_modelo"]["roi"]:.1f}',
            "con_pin n/roi%": f'{g["con_pinnacle"]["n"]}/{100 * g["con_pinnacle"]["roi"]:.1f}',
            "beats_pm": s["beats_pm"], "beats_book": s["beats_book"],
            "n_train 1º/últ": f'{s["blocks"][0]["n_train"]}/{s["blocks"][-1]["n_train"]}' if s["blocks"] else "-",
        })
    return pd.DataFrame(rows)


def load_spread_table(path):
    return tuple(tuple(x) for x in json.load(open(path))["table"])
