"""Fuentes de cuotas gratuitas.
- Histórico: football-data.co.uk (CSV gratis con cierre de Pinnacle: PSCH/PSCD/PSCA).
- En vivo: The Odds API (plan gratis ~500 req/mes), bookmaker 'pinnacle' en region 'eu'.
"""
import os
import numpy as np
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


LEAGUES = ("E0", "SP1", "D1", "I1", "F1")  # solo primeras divisiones: en segundas la cobertura de PM es marginal


def check_scope(track: str, seasons, leagues):
    """Pista A solo 25/26 (26/27 no tiene PSH). Ligas solo de LEAGUES."""
    bad = [l for l in leagues if l not in LEAGUES]
    if bad:
        raise SystemExit(f"Ligas no permitidas {bad}: solo {list(LEAGUES)} (sin segundas divisiones).")
    if track == "A" and tuple(seasons) != ("2526",):
        raise SystemExit("La pista A es solo --seasons 2526: 26/27 no tiene PSH y no aporta filas.")


def resolve_fd(paths, leagues=LEAGUES):
    """--fd acepta ficheros o una carpeta. En una carpeta busca un CSV por liga ({liga}.csv o que
    empiece por la liga, p. ej. 'E0 (1).csv' de una descarga del navegador). Falla si falta alguna."""
    import glob
    import os
    out = []
    for p in paths:
        if not os.path.isdir(p):
            out.append(p)
            continue
        files = sorted(glob.glob(os.path.join(p, "*.csv")))
        missing = []
        for lg in leagues:
            hit = [f for f in files if os.path.basename(f).upper().startswith(lg.upper())
                   and not os.path.basename(f).upper().startswith(lg.upper() + "_")]
            exact = [f for f in hit if os.path.basename(f).upper() == f"{lg.upper()}.CSV"]
            (out.append((exact or hit)[0]) if hit else missing.append(lg))
        if missing:
            raise SystemExit(f"{p}: faltan CSV de {missing}. Descárgalos de football-data (temporada 2526).")
    return out


def fd_urls(seasons=("2526",), leagues=LEAGUES):
    return [FD_URL.format(season=s, league=l) for s in seasons for l in leagues]


def _cols(book):
    return {f"{book}CH": "oH", f"{book}CD": "oD", f"{book}CA": "oA",
            f"{book}H": "ooH", f"{book}D": "ooD", f"{book}A": "ooA"}


BASE_COLS = ("Date", "HomeTeam", "AwayTeam", "FTR")


def _pattern(dates_sorted, dropped):
    """'bloque' si las filas descartadas son contiguas en el calendario del fichero, si no 'goteo'."""
    idx = np.flatnonzero(dropped)
    if len(idx) == 0:
        return "-"
    contiguous = idx[-1] - idx[0] + 1 == len(idx)
    if contiguous and idx[-1] == len(dropped) - 1:
        return f"bloque final desde {dates_sorted[idx[0]]}"
    return f"bloque {dates_sorted[idx[0]]}..{dates_sorted[idx[-1]]}" if contiguous else "goteo"


def load_football_data(srcs, book="PS", until=None, log=print) -> pd.DataFrame:
    """Carga UNA casa y distingue dos casos:
    - COLUMNA AUSENTE en un fichero (p. ej. PSH en 26/27): aviso con el nombre del fichero, la
      temporada se trunca en la fecha del último partido con datos y se continúa. Solo se aborta si
      la columna falta en TODOS los ficheros, con mensaje de columna ausente.
    - VALOR VACÍO: se descarta la fila y se cuenta; se registran sus fechas por fichero.
    El registro de descartes queda en df.attrs['drops'] (una fila por fichero)."""
    need = list(_cols(book))
    out, drops, absent = [], [], []
    for src in ([srcs] if isinstance(srcs, str) else srcs):
        df = pd.read_csv(src, encoding="latin-1")
        miss = [c for c in (*BASE_COLS, *need) if c not in df.columns]
        if miss:
            absent.append((src, miss))
            log(f"AVISO {src}: columna ausente {miss}. Fichero omitido; la temporada se trunca "
                f"en el último partido con datos.")
            continue
        df = df.rename(columns=_cols(book))
        df["date"] = pd.to_datetime(df["Date"], dayfirst=True).dt.date
        df["league"] = df["Div"] if "Div" in df else "?"
        df = df.dropna(subset=["FTR"]).sort_values("date").reset_index(drop=True)
        n_cut = 0
        if until:
            keep = df.date < pd.Timestamp(until).date()
            n_cut, df = int((~keep).sum()), df[keep].reset_index(drop=True)
        no_open = df[["ooH", "ooD", "ooA"]].isna().any(axis=1).values
        no_close = df[["oH", "oD", "oA"]].isna().any(axis=1).values & ~no_open
        dropped = no_open | no_close
        dates = df.date.astype(str).tolist()
        drops.append({"file": src, "rows": len(df), "cut_by_date": n_cut,
                      "dropped_no_open": int(no_open.sum()), "dropped_no_close": int(no_close.sum()),
                      "pattern": _pattern(dates, dropped),
                      "dates": ",".join(sorted(set(np.array(dates)[dropped])))})
        log(f"{src}: {len(df)} partidos, descartados {int(no_open.sum())} sin apertura {book} y "
            f"{int(no_close.sum())} sin cierre {book} ({drops[-1]['pattern']})"
            + (f"; {n_cut} posteriores a {until} fuera por corte" if n_cut else ""))
        out.append(df.loc[~dropped, ["date", "league", "HomeTeam", "AwayTeam", "FTR", *_cols(book).values()]])
    if not out:
        cols = sorted({c for _, m in absent for c in m})
        raise SystemExit(f"ABORTADO: columna ausente en todos los ficheros: {cols} ({BOOKS.get(book, book)}). "
                         "No se cambia de casa.")
    df = pd.concat(out)
    if df.empty:
        raise SystemExit(f"ABORTADO: las columnas {book}* existen pero todas las filas están vacías.")
    if absent:
        log(f"Temporada truncada en {df.date.max()} (último partido con datos {book}).")
    df["source"] = book
    df.attrs["drops"] = pd.DataFrame(drops)
    df.attrs["absent"] = [{"file": f, "missing": ",".join(m)} for f, m in absent]
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
