"""Fuentes de cuotas gratuitas.
- Histórico: football-data.co.uk (CSV gratis con cierre de Pinnacle: PSCH/PSCD/PSCA).
- En vivo: The Odds API (plan gratis ~500 req/mes), bookmaker 'pinnacle' en region 'eu'.
"""
import os
import pandas as pd
from .http import get_json


FD_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
BOOKS = {"PS": "Pinnacle", "BFE": "Betfair Exchange", "Avg": "media del mercado", "Max": "máximo del mercado"}
# Fallback: Avg antes que Max. Max solo como último recurso: es un outlier por construcción e infla el edge.
FALLBACK = ("Avg", "Max")

# Pistas separadas. Nunca se mezclan fuentes en un mismo backtest.
TRACKS = {
    "A": {"seasons": ("2526",), "book": "PS", "until": "2026-01-08",
          "note": "Pinnacle hasta el 08/01/2026 (desaparece de football-data). Único veredicto válido."},
    "B": {"seasons": ("2627",), "book": "BFE", "until": None,
          "note": "Betfair Exchange como referencia; 26/27 no trae Pinnacle."},
}

# Horizonte de las cuotas de football-data:
#   apertura ({book}H/D/A): recogida viernes y martes por la tarde -> T-24h a T-72h del saque.
#   cierre   ({book}CH/CD/CA): al saque.
OPEN_HORIZON_H = (24, 72)
DECISION_H = 24  # punto de decisión con la línea de apertura (backtest y escáner)


def fd_urls(seasons=("2526",), leagues=("E0", "SP1", "I1", "D1", "F1")):
    return [FD_URL.format(season=s, league=l) for s in seasons for l in leagues]


def _cols(book):
    return {f"{book}CH": "oH", f"{book}CD": "oD", f"{book}CA": "oA",
            f"{book}H": "ooH", f"{book}D": "ooD", f"{book}A": "ooA"}


def _has(df, book):
    return f"{book}H" in df and df[f"{book}H"].notna().any()


def choose_book(raws: list, book: str, log=print) -> str:
    """Una sola fuente para TODOS los ficheros: la pedida o, si falta, Avg y luego Max."""
    for b in (book, *[x for x in FALLBACK if x != book]):
        if all(_has(df, b) for df in raws):
            if b != book:
                log(f"AVISO: sin {BOOKS.get(book, book)} en todos los ficheros; uso {BOOKS[b]} ({b}).")
            if b == "Max":
                log("AVISO: Max es un outlier por construcción e infla el edge. Resultado no válido como veredicto.")
            return b
    raise SystemExit(f"Ninguna fuente ({book}, {', '.join(FALLBACK)}) está en todos los ficheros. No se mezclan fuentes.")


def load_football_data(srcs, book="PS", until=None, log=print) -> pd.DataFrame:
    raws = [pd.read_csv(s, encoding="latin-1") for s in ([srcs] if isinstance(srcs, str) else srcs)]
    b = choose_book(raws, book, log)
    out = []
    for df in raws:
        ren = _cols(b)
        df = df.rename(columns=ren)
        for c in ren.values():
            if c not in df:
                df[c] = float("nan")
        df["date"] = pd.to_datetime(df["Date"], dayfirst=True).dt.date
        df["league"] = df["Div"] if "Div" in df else "?"
        out.append(df[["date", "league", "HomeTeam", "AwayTeam", "FTR", *ren.values()]])
    df = pd.concat(out).dropna(subset=["FTR", "ooH", "ooD", "ooA"])
    if until:
        df = df[df.date < pd.Timestamp(until).date()]
    df["source"] = b
    log(f"{len(df)} partidos con cuotas {b}" + (f" anteriores a {until}" if until else ""))
    return df


def odds_api_live(sport="soccer_epl", bookmaker="pinnacle", api_key=None):
    key = api_key or os.environ["ODDS_API_KEY"]
    data = get_json(f"https://api.the-odds-api.com/v4/sports/{sport}/odds",
                    {"apiKey": key, "regions": "eu", "markets": "h2h", "oddsFormat": "decimal",
                     "bookmakers": bookmaker})
    rows = []
    for ev in data:
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                o = {x["name"]: x["price"] for x in mk["outcomes"]}
                rows.append({"start": ev["commence_time"], "home": ev["home_team"], "away": ev["away_team"],
                             "oH": o.get(ev["home_team"]), "oD": o.get("Draw"), "oA": o.get(ev["away_team"])})
    return pd.DataFrame(rows)
