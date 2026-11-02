"""Emparejar partidos de la casa con eventos/mercados de Polymarket."""
import difflib
import re
import unicodedata

STOP = r"\b(fc|cf|afc|sc|ac|cd|ud|rcd|sd|club|de|the|calcio|ssc|as|1\.)\b"
ALIASES = {"man united": "manchester united", "man city": "manchester city", "spurs": "tottenham",
           "nott m forest": "nottingham forest", "wolves": "wolverhampton", "ath madrid": "atletico madrid",
           "ath bilbao": "athletic bilbao", "betis": "real betis", "sociedad": "real sociedad"}


def norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 .]", " ", s)
    s = ALIASES.get(s.strip(), s)
    s = re.sub(STOP, " ", s)
    return re.sub(r"\s+", " ", s).strip()


def sim(a: str, b: str) -> float:
    a, b = norm(a), norm(b)
    if a and b and (a in b or b in a):
        return 0.95
    return difflib.SequenceMatcher(None, a, b).ratio()


def split_title(title: str):
    parts = re.split(r"\s+vs\.?\s+", title, flags=re.I)
    return (parts[0], parts[1]) if len(parts) == 2 else None


def outcome_of(question: str, home: str, away: str):
    """Mapea la pregunta binaria del mercado a H / D / A."""
    if re.search(r"\bdraw\b|empate", question, re.I):
        return "D"
    sh, sa = sim_in(question, home), sim_in(question, away)
    return "H" if sh > sa else "A"


def sim_in(text: str, team: str) -> float:
    t, n = norm(text), norm(team)
    return 1.0 if n and n in t else max((difflib.SequenceMatcher(None, n, w).ratio()
                                         for w in t.split()), default=0)


def match_event(home: str, away: str, candidates: list[dict], min_score=0.75):
    """candidates: eventos de Polymarket con 'title'. Devuelve el mejor o None."""
    best, best_s = None, 0.0
    for ev in candidates:
        sp = split_title(ev.get("title", ""))
        if not sp:
            continue
        s = min(sim(home, sp[0]), sim(away, sp[1]))
        if s > best_s:
            best, best_s = ev, s
    return best if best_s >= min_score else None
