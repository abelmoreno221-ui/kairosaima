"""Emparejar partidos de la casa con eventos/mercados de Polymarket."""
import difflib
import re
import unicodedata

STOP = r"\b(fc|cf|afc|sc|ac|cd|ud|rcd|sd|club|de|the|calcio|ssc|as|1\.)\b"
ALIASES = {"man united": "manchester united", "man city": "manchester city", "spurs": "tottenham",
           "nott m forest": "nottingham forest", "wolves": "wolverhampton", "ath madrid": "atletico madrid",
           "ath bilbao": "athletic bilbao", "betis": "real betis", "sociedad": "real sociedad",
           "inter": "internazionale", "internazionale milano": "internazionale",
           "wolverhampton wanderers": "wolverhampton"}


def norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 .]", " ", s)
    s = ALIASES.get(s.strip(), s)
    s = re.sub(STOP, " ", s)
    return re.sub(r"\s+", " ", s).strip()


def sim(a: str, b: str) -> float:
    """Similitud por PALABRAS completas ('milan' no casa con 'milano'), luego difflib."""
    a, b = norm(a), norm(b)
    wa, wb = set(a.split()), set(b.split())
    if wa and wb and (wa <= wb or wb <= wa):
        return 0.95                                   # palabras exactas
    short, long_ = (wa, wb) if len(wa) <= len(wb) else (wb, wa)
    if short and all(len(w) >= 4 and any(x.startswith(w) or w.startswith(x) for x in long_) for w in short):
        return 0.90                                   # prefijo: hamburg~hamburger, lyon~lyonnais
    return difflib.SequenceMatcher(None, a, b).ratio()


WIN_RE = re.compile(r"^\s*will\s+(.+?)\s+win\b", re.I)


def is_1x2(ev: dict) -> bool:
    """Evento de resultado (local/empate/visitante), no hándicap ni más/menos goles."""
    qs = [m.get("question", "") for m in ev.get("markets", [])]
    return any(WIN_RE.search(q) for q in qs) and any(re.search(r"\bdraw\b", q, re.I) for q in qs)


def split_title(title: str):
    parts = re.split(r"\s+vs\.?\s+", title, flags=re.I)
    return (parts[0], parts[1]) if len(parts) == 2 else None


def outcome_of(question: str, home: str, away: str):
    """Mapea la pregunta binaria del mercado a H / D / A."""
    if re.search(r"\bdraw\b|empate", question, re.I):
        return "D"
    m = WIN_RE.search(question)
    if m:
        sh, sa = sim(m.group(1), home), sim(m.group(1), away)
    else:
        sh, sa = sim_in(question, home), sim_in(question, away)
    if sh == sa:
        return None                      # ambiguo: no se adivina
    return "H" if sh > sa else "A"


def sim_in(text: str, team: str) -> float:
    t, n = norm(text), norm(team)
    return 1.0 if n and n in t else max((difflib.SequenceMatcher(None, n, w).ratio()
                                         for w in t.split()), default=0)


def match_event(home: str, away: str, candidates: list[dict], min_score=0.75):
    """candidates: eventos de Polymarket con 'title'. Devuelve el mejor o None."""
    best, best_s = None, 0.0
    for ev in candidates:
        if ev.get("markets") and not is_1x2(ev):
            continue
        sp = split_title(ev.get("title", ""))
        if not sp:
            continue
        s = min(sim(home, sp[0]), sim(away, sp[1]))
        if s > best_s:
            best, best_s = ev, s
    return best if best_s >= min_score else None
