"""Cliente de solo lectura para Polymarket (Gamma API + CLOB público). Sin claves, sin ejecución."""
import json
from .http import get_json

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"


def _j(x):
    return json.loads(x) if isinstance(x, str) else (x or [])


def events(tag_slug="soccer", closed=False, limit=100, offset=0):
    return get_json(f"{GAMMA}/events", {"tag_slug": tag_slug, "closed": str(closed).lower(),
                                         "limit": limit, "offset": offset})


def iter_events(tag_slug="soccer", closed=False, max_pages=50, page=100):
    for i in range(max_pages):
        batch = events(tag_slug, closed, page, i * page)
        if not batch:
            return
        yield from batch


def parse_market(m: dict) -> dict:
    outcomes, tokens, prices = _j(m.get("outcomes")), _j(m.get("clobTokenIds")), _j(m.get("outcomePrices"))
    yes = outcomes.index("Yes") if "Yes" in outcomes else 0
    return {
        "id": m.get("id"), "question": m.get("question", ""), "condition_id": m.get("conditionId"),
        "yes_token": tokens[yes] if tokens else None,
        "no_token": tokens[1 - yes] if len(tokens) == 2 else None,
        "yes_price": float(prices[yes]) if prices else None,
        "start": m.get("gameStartTime") or m.get("eventStartTime"), "end": m.get("endDate"),
        "closed": m.get("closed"), "fee_bps": m.get("takerBaseFee") or m.get("feeRate"),
    }


def price_history(token_id: str, start_ts: int, end_ts: int, fidelity_min=5):
    """Serie [(ts, precio)] del token YES."""
    r = get_json(f"{CLOB}/prices-history", {"market": token_id, "startTs": start_ts,
                                             "endTs": end_ts, "fidelity": fidelity_min})
    return [(int(h["t"]), float(h["p"])) for h in r.get("history", [])]


def price_at(token_id: str, ts: int, window_s=6 * 3600):
    """Último precio conocido <= ts (sin lookahead)."""
    hist = [h for h in price_history(token_id, ts - window_s, ts) if h[0] <= ts]
    return hist[-1][1] if hist else None


def prices_at(token_id: str, ts_list, window_s=6 * 3600):
    """Último precio <= cada ts, con UNA sola llamada a prices-history. None si no hay."""
    hist = price_history(token_id, min(ts_list) - window_s, max(ts_list))
    out = []
    for ts in ts_list:
        h = [p for t, p in hist if ts - window_s <= t <= ts]
        out.append(h[-1] if h else None)
    return out


def top_of_book(token_id: str):
    b = get_json(f"{CLOB}/book", {"token_id": token_id})
    bids = sorted((float(x["price"]), float(x["size"])) for x in b.get("bids", []))
    asks = sorted((float(x["price"]), float(x["size"])) for x in b.get("asks", []))
    return {"bid": bids[-1] if bids else None, "ask": asks[0] if asks else None}


# ---------- Data API: trades (liquidez sin lookahead) ----------
import hashlib
import os
import time as _time

DATA = "https://data-api.polymarket.com"
CACHE_DIR = os.environ.get("PMARB_CACHE", ".cache")
_MIN_INTERVAL = float(os.environ.get("PMARB_MIN_INTERVAL", "0.25"))  # s entre llamadas (~4 req/s)
_last = [0.0]


def _throttle():
    wait = _MIN_INTERVAL - (_time.monotonic() - _last[0])
    if wait > 0:
        _time.sleep(wait)
    _last[0] = _time.monotonic()


def trades(condition_id: str, until_ts: int, page=500, max_offset=10000):
    """Trades de un mercado. /trades devuelve lo MÁS RECIENTE primero, así que se pagina con offset
    hasta encontrar un trade con t < until_ts (ya se tiene todo lo posterior) o hasta agotar el
    histórico. Si se agota el offset antes, 'capped' = True -> volumen desconocido.
    Cacheado por mercado; solo se vuelve a pedir si la caché no llega tan atrás y no está capada."""
    path = os.path.join(CACHE_DIR, "trades", hashlib.sha1(condition_id.encode()).hexdigest() + ".json")
    c = json.load(open(path)) if os.path.exists(path) else None
    if c and (c["exhausted"] or c["capped"] or c["oldest"] < until_ts):
        return c
    out, exhausted, capped = [], False, False
    for off in range(0, max_offset, page):
        _throttle()
        batch = get_json(f"{DATA}/trades", {"market": condition_id, "limit": page, "offset": off})
        out += [{"t": int(x["timestamp"]), "usd": float(x["size"]) * float(x["price"])} for x in batch]
        if len(batch) < page:
            exhausted = True
            break
        if min(x["t"] for x in out) < until_ts:
            break
    else:
        capped = True
    c = {"trades": out, "oldest": min((x["t"] for x in out), default=2 ** 62),
         "exhausted": exhausted, "capped": capped and not exhausted}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(c, open(path, "w"))
    return c


def volume_window(c: dict, ts: int, window_s: int):
    """USDC negociados en [ts - window, ts). None si la caché no cubre ese tramo (desconocido)."""
    lo = ts - window_s
    if not (c["exhausted"] or c["oldest"] < lo):
        return None
    return float(sum(x["usd"] for x in c["trades"] if lo <= x["t"] < ts))


def spread(token_id: str):
    _throttle()
    return float(get_json(f"{CLOB}/spread", {"token_id": token_id}).get("spread", "nan"))


def book_snapshot(token_id: str, depth_cents=0.02):
    _throttle()
    b = get_json(f"{CLOB}/book", {"token_id": token_id})
    bids = sorted(((float(x["price"]), float(x["size"])) for x in b.get("bids", [])), reverse=True)
    asks = sorted((float(x["price"]), float(x["size"])) for x in b.get("asks", []))
    if not bids or not asks:
        return None
    bb, ba = bids[0][0], asks[0][0]
    return {"bid": bb, "ask": ba, "mid": (bb + ba) / 2, "book_spread": ba - bb,
            "bid_size": bids[0][1], "ask_size": asks[0][1],
            "ask_depth_usd": sum(p * s for p, s in asks if p <= ba + depth_cents)}
