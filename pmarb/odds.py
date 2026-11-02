"""Fuentes de cuotas gratuitas.
- Histórico: football-data.co.uk (CSV gratis con cierre de Pinnacle: PSCH/PSCD/PSCA).
- En vivo: The Odds API (plan gratis ~500 req/mes), bookmaker 'pinnacle' en region 'eu'.
"""
import os
import pandas as pd
from .http import get_json


FD_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
BOOKS = {"PS": "Pinnacle", "BFE": "Betfair Exchange", "Avg": "media del mercado", "Max": "máximo del mercado"}

# Una pista = una casa. Nunca se cambia de casa dentro de una pista: las filas sin cuotas
# de esa casa (apertura o cierre) se DESCARTAN.
TRACKS = {
    "A": {"seasons": ("2526",), "book": "PS", "until": "2026-01-08", "valid": True,
          "note": "Pinnacle hasta el 08/01/2026 (desaparece de football-data). Único veredicto válido."},
    "B": {"seasons": ("2627",), "book": "BFE", "until": None, "valid": False,
          "note": "Betfair Exchange como referencia (exchange: comisión sobre ganancia neta)."},
    "C": {"seasons": ("2526", "2627"), "book": "Avg", "until": None, "valid": False,
          "note": "Avg (o Max con --book Max). INVÁLIDA: solo diagnóstico. Max infla el edge por construcción."},
}
TRACK_C_BOOKS = ("Avg", "Max")

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


def load_football_data(srcs, book="PS", until=None, log=print) -> pd.DataFrame:
    """Carga UNA casa. Filas sin las 6 cuotas de esa casa (apertura + cierre) se descartan y se cuentan."""
    out = []
    for src in ([srcs] if isinstance(srcs, str) else srcs):
        df = pd.read_csv(src, encoding="latin-1")
        ren = _cols(book)
        df = df.rename(columns=ren)
        for c in ren.values():
            if c not in df:
                df[c] = float("nan")
        df["date"] = pd.to_datetime(df["Date"], dayfirst=True).dt.date
        df["league"] = df["Div"] if "Div" in df else "?"
        df = df.dropna(subset=["FTR"])
        if until:
            df = df[df.date < pd.Timestamp(until).date()]
        n0 = len(df)
        no_open = df[["ooH", "ooD", "ooA"]].isna().any(axis=1)
        no_close = df[["oH", "oD", "oA"]].isna().any(axis=1) & ~no_open
        df = df[~(no_open | no_close)]
        log(f"{src}: {n0} partidos, descartados {int(no_open.sum())} sin apertura {book} "
            f"y {int(no_close.sum())} sin cierre {book} -> {len(df)}")
        out.append(df[["date", "league", "HomeTeam", "AwayTeam", "FTR", *ren.values()]])
    df = pd.concat(out)
    if df.empty:
        raise SystemExit(f"ABORTADO: ningún partido con cuotas {BOOKS.get(book, book)} ({book}*). No se cambia de casa.")
    df["source"] = book
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
