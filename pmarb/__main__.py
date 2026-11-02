import argparse
import json
from dataclasses import replace
import pandas as pd
from .backtest import Cfg, grid, table, with_book, load_spread_table, SPREADS
from .model import Blend


def main():
    ap = argparse.ArgumentParser("pmarb")
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build", help="dataset football-data x Polymarket (aborta si la cobertura es baja)")
    b.add_argument("--fd", nargs="*", help="CSV locales; si no, descarga --seasons x --leagues")
    b.add_argument("--seasons", nargs="+", default=["2526", "2627"])
    b.add_argument("--leagues", nargs="+", default=["E0", "SP1", "I1", "D1", "F1"])
    b.add_argument("--book", default="PS", help="PS (Pinnacle), Avg, Max, B365")
    b.add_argument("--tag", default="soccer")
    b.add_argument("--lead-min", type=int, default=10)
    b.add_argument("--min-coverage", type=float, default=0.30)
    b.add_argument("--min-matches", type=int, default=150)
    b.add_argument("--out", default="dataset.csv")
    s = sp.add_parser("synthetic")
    s.add_argument("--out", default="dataset.csv")
    t = sp.add_parser("backtest")
    t.add_argument("data")
    t.add_argument("--fee-rate", type=float, default=0.05)
    t.add_argument("--min-edge", type=float, default=0.04)
    t.add_argument("--min-volume", type=float, default=1000.0, help="USDC negociados antes de la señal")
    t.add_argument("--half-spreads", type=float, nargs="+", default=list(SPREADS), help="MEDIO spread en $")
    t.add_argument("--spread-table", help="spread_table.json de `spreads summary` (añade nivel 'cal')")
    t.add_argument("--min-matches", type=int, default=150)
    t.add_argument("--json-out", default="backtest.json")
    f = sp.add_parser("fit")
    f.add_argument("data")
    f.add_argument("--out", default="weights.json")
    sc = sp.add_parser("spreads", help="muestrear /spread y /book en fútbol vivo")
    sc.add_argument("action", choices=["sample", "summary"])
    sc.add_argument("--csv", default="spreads.csv")
    sc.add_argument("--out", default="spread_table.json")
    sc.add_argument("--quantile", type=float, default=0.5)
    c = sp.add_parser("scan", help="señales en papel (requiere ODDS_API_KEY)")
    c.add_argument("--sport", default="soccer_epl")
    c.add_argument("--tag", default="soccer")
    a = ap.parse_args()

    if a.cmd == "build":
        from .dataset import build
        from .odds import load_football_data, fd_urls
        srcs = a.fd or fd_urls(a.seasons, a.leagues)
        fd = pd.concat([load_football_data(p, a.book) for p in srcs])
        build(fd, a.tag, a.lead_min, min_rate=a.min_coverage, min_matches=a.min_matches).to_csv(a.out, index=False)
    elif a.cmd == "synthetic":
        from .dataset import synthetic
        synthetic().to_csv(a.out, index=False)
    elif a.cmd == "backtest":
        df = pd.read_csv(a.data)
        if df.match_id.nunique() < a.min_matches:
            raise SystemExit(f"ABORTADO: {df.match_id.nunique()} partidos < {a.min_matches}. No se calculan cifras.")
        cfg = Cfg(fee_rate=a.fee_rate, min_edge=a.min_edge, min_volume=a.min_volume,
                  spread_table=load_spread_table(a.spread_table) if a.spread_table else None)
        res, v = grid(df, cfg, spreads=tuple(a.half_spreads))
        with pd.option_context("display.width", 300, "display.max_columns", 40):
            print("half_spr = MEDIO spread ($/acción), pagado al entrar como taker\n")
            print(table(res).to_string(index=False))
        g = v.pop("guardrail")
        print(json.dumps(v, indent=1, ensure_ascii=False))
        print("\nGuardarraíl Brier(modelo) - Brier(casa), IC95 bootstrap por partido (>0 = modelo peor):")
        for k, x in g.items():
            flag = "  <-- AVISO: peor que la casa" if x["model_worse_than_book"] else ""
            print(f"  {k:12s} {x['brier_diff']:+.5f}  [{x['ci95'][0]:+.5f}, {x['ci95'][1]:+.5f}]{flag}")
        v["guardrail"] = g
        seen = {}
        for (l, m, n), s in res.items():
            seen.setdefault(tuple(x["n_train"] for x in s["blocks"]), set()).add(f"{l}/{m}")
        for sizes, who in seen.items():
            print(f"n_train por bloque ({', '.join(sorted(who))}):", " ".join(map(str, sizes)))
        json.dump({"/".join(k): s for k, s in res.items()} | {"verdict": v},
                  open(a.json_out, "w"), indent=2, default=str)
    elif a.cmd == "fit":
        df = with_book(pd.read_csv(a.data), "open", "shin")
        m = Blend().fit(df.p_pm.values, df.p_book.values, df.y.values)
        json.dump({"w": m.w.tolist(), "n": len(df)}, open(a.out, "w"))
        print("w =", m.w.round(3), "(intercepto, peso PM, peso casa)")
    elif a.cmd == "spreads":
        from . import spreads
        if a.action == "sample":
            spreads.sample(out=a.csv)
        else:
            st, _ = spreads.summary(a.csv, a.out, a.quantile)
            print(st.to_string())
    elif a.cmd == "scan":
        from .scan import scan
        print(scan(a.sport, a.tag).to_string())


if __name__ == "__main__":
    main()
