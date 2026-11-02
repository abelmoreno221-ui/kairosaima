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


def top_of_book(token_id: str):
    b = get_json(f"{CLOB}/book", {"token_id": token_id})
    bids = sorted((float(x["price"]), float(x["size"])) for x in b.get("bids", []))
    asks = sorted((float(x["price"]), float(x["size"])) for x in b.get("asks", []))
    return {"bid": bids[-1] if bids else None, "ask": asks[0] if asks else None}
