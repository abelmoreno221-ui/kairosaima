"""Fuentes de cuotas gratuitas.
- Histórico: football-data.co.uk (CSV gratis con cierre de Pinnacle: PSCH/PSCD/PSCA).
- En vivo: The Odds API (plan gratis ~500 req/mes), bookmaker 'pinnacle' en region 'eu'.
"""
import os
import pandas as pd
from .http import get_json


def load_football_data(path_or_url: str) -> pd.DataFrame:
    """Devuelve cuotas de cierre (oH/oD/oA <- PSCH/PSCD/PSCA) y apertura (ooH/ooD/ooA <- PSH/PSD/PSA).
    La apertura es el proxy de lo disponible a ~T-6h; el cierre no existe en el momento de decidir."""
    df = pd.read_csv(path_or_url, encoding="latin-1")
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True).dt.date
    ren = {"PSCH": "oH", "PSCD": "oD", "PSCA": "oA", "PSH": "ooH", "PSD": "ooD", "PSA": "ooA"}
    df = df.rename(columns=ren)
    for c in ren.values():
        if c not in df:
            df[c] = float("nan")
    return df[["date", "HomeTeam", "AwayTeam", "FTR", *ren.values()]].dropna(subset=["FTR"])


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
