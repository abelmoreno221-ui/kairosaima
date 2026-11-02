"""Fuentes de cuotas gratuitas.
- Histórico: football-data.co.uk (CSV gratis con cierre de Pinnacle: PSCH/PSCD/PSCA).
- En vivo: The Odds API (plan gratis ~500 req/mes), bookmaker 'pinnacle' en region 'eu'.
"""
import os
import pandas as pd
from .http import get_json


FD_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
BOOKS = {"PS": "Pinnacle", "Avg": "media del mercado", "Max": "máximo del mercado", "B365": "Bet365"}


def fd_urls(seasons=("2526", "2627"), leagues=("E0", "SP1", "I1", "D1", "F1")):
    return [FD_URL.format(season=s, league=l) for s in seasons for l in leagues]


def load_football_data(path_or_url: str, book="PS") -> pd.DataFrame:
    """Cierre (oH/oD/oA <- {book}CH/CD/CA) y apertura (ooH/ooD/ooA <- {book}H/D/A).
    La apertura es el proxy de lo disponible antes del partido; el cierre no existe al decidir.
    Si el fichero no trae columnas de esa casa, falla con las alternativas disponibles."""
    df = pd.read_csv(path_or_url, encoding="latin-1")
    ren = {f"{book}CH": "oH", f"{book}CD": "oD", f"{book}CA": "oA",
           f"{book}H": "ooH", f"{book}D": "ooD", f"{book}A": "ooA"}
    have = [c for c in ren if c in df]
    if not have or df[have].notna().sum().sum() == 0:
        alt = [b for b in BOOKS if f"{b}H" in df and df[f"{b}H"].notna().any()]
        raise SystemExit(f"{path_or_url}: sin cuotas de {BOOKS.get(book, book)} ({book}*). "
                         f"Disponibles: {alt}. Usa --book con una de ellas.")
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True).dt.date
    df = df.rename(columns=ren)
    for c in ren.values():
        if c not in df:
            df[c] = float("nan")
    df["league"] = df.get("Div", "?")
    return df[["date", "league", "HomeTeam", "AwayTeam", "FTR", *ren.values()]].dropna(subset=["FTR"])


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
