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
    b.add_argument("--track", choices=["A", "B", "C"], required=True,
                   help="A: 25/26 Pinnacle hasta 08/01/2026 (veredicto). B: 26/27 Betfair Exchange (referencia). "
                        "C: Avg/Max (inválida, diagnóstico)")
    b.add_argument("--book", choices=["Avg", "Max"], help="solo pista C (por defecto Avg)")
    b.add_argument("--seasons", nargs="+", help="sobrescribe las temporadas de la pista (p. ej. 2526 2627)")
    b.add_argument("--fd", nargs="*", help="CSV locales de la temporada de la pista; si no, se descargan")
    b.add_argument("--leagues", nargs="+", default=["E0", "SP1", "D1", "I1", "F1"])
    b.add_argument("--tag", default="soccer")
    b.add_argument("--vol-window-h", type=int, default=168, help="ventana de volumen previo a cada decisión")
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
    t.add_argument("--coverage", help="cobertura por liga de `build` (por defecto <data>.coverage.csv)")
    t.add_argument("--decision-h", choices=["24", "6", "close"], default="24",
                   help="24: validación conservadora; 6: operativo; close: T-10min, techo de investigación")
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
        from .odds import load_football_data, fd_urls, TRACKS, check_scope
        tr = TRACKS[a.track]
        print(f"Pista {a.track}: {tr['note']}")
        if a.book and a.track != "C":
            raise SystemExit("--book solo vale para la pista C: las pistas A y B no cambian de casa.")
        book = a.book or tr["book"]
        seasons = a.seasons or tr["seasons"]
        check_scope(a.track, seasons, a.leagues)
        fd = load_football_data(a.fd or fd_urls(seasons, a.leagues), book, tr["until"])
        out = a.out if a.out != "dataset.csv" else f"dataset_{a.track}.csv"
        drops = fd.attrs["drops"]
        if fd.attrs["absent"]:
            drops = pd.concat([drops, pd.DataFrame([{**x, "pattern": "columna ausente"} for x in fd.attrs["absent"]])])
        drops.to_csv(out.replace(".csv", ".drops.csv"), index=False)
        ds, cov = build(fd, a.tag, vol_window_h=a.vol_window_h, min_rate=a.min_coverage, min_matches=a.min_matches)
        ds.to_csv(out, index=False)
        cov.to_csv(out.replace(".csv", ".coverage.csv"), index=False)
        print("\n=== PASO 1: leer solo esto ===")
        print(cov.to_string(index=False))
        print(f"TOTAL emparejados: {int(cov.matched.sum())} (mínimo 150 para seguir al backtest)")
        for _, r in drops.fillna("").iterrows():
            if r.pattern == "columna ausente":
                print(f"  {r.file}: COLUMNA AUSENTE {r.missing}")
            else:
                print(f"  {r.file}: {int(r.dropped_no_open)} sin apertura, {int(r.dropped_no_close)} sin cierre, "
                      f"{int(r.cut_by_date)} fuera por corte de fecha -> {r.pattern}")
        print(f"-> {out}")
    elif a.cmd == "synthetic":
        from .dataset import synthetic
        synthetic().to_csv(a.out, index=False)
    elif a.cmd == "backtest":
        df = pd.read_csv(a.data)
        if df.match_id.nunique() < a.min_matches:
            raise SystemExit(f"ABORTADO: {df.match_id.nunique()} partidos < {a.min_matches}. No se calculan cifras.")
        cfg = Cfg(fee_rate=a.fee_rate, min_edge=a.min_edge, min_volume=a.min_volume,
                  spread_table=load_spread_table(a.spread_table) if a.spread_table else None)
        import os
        cov_path = a.coverage or a.data.replace(".csv", ".coverage.csv")
        cov = pd.read_csv(cov_path) if os.path.exists(cov_path) else None
        res, v = grid(df, cfg, spreads=tuple(a.half_spreads), coverage=cov, decision=a.decision_h)
        with pd.option_context("display.width", 300, "display.max_columns", 40):
            print("half_spr = MEDIO spread ($/acción), pagado al entrar como taker")
            for k, h in v["horizons"].items():
                print(f"horizonte {k}: {h}")
            print(f"fuente: {v['source']}\n")
            print(table(res).to_string(index=False))
        unk = [(k, s["unknown_volume"]) for k, s in res.items() if s.get("unknown_volume", {}).get("markets")]
        if unk:
            print("\nVolumen DESCONOCIDO (/trades agotó el offset antes del punto de decisión). Fuera de las cifras de arriba:")
            for (l, m, n), u in unk:
                print(f"  {l}/{m}@{n}: {u['markets']} filas, {u['signals']} señales ({u['signals_per_week']:.2f}/sem), ROI {100 * u['roi']:.1f}%")
        dp = a.data.replace(".csv", ".drops.csv")
        if os.path.exists(dp):
            dr = pd.read_csv(dp).fillna("")
            print("\nFILAS DESCARTADAS POR FICHERO (bloque = corte de la fuente; goteo = huecos sueltos):")
            for _, r in dr.iterrows():
                if r.get("pattern") == "columna ausente":
                    print(f"  {r.file}: COLUMNA AUSENTE {r.missing} -> fichero omitido, temporada truncada")
                    continue
                n = int(r.dropped_no_open) + int(r.dropped_no_close)
                print(f"  {r.file}: {n} descartadas ({int(r.dropped_no_open)} sin apertura, "
                      f"{int(r.dropped_no_close)} sin cierre) -> {r.pattern}")
                if n:
                    print(f"     fechas: {r.dates}")
        fn = pd.DataFrame(v["funnel"])
        print(f"\nEMBUDO POR LIGA ({v['funnel_level']}; señal neta solo en periodo fuera de muestra):")
        print(fn.to_string(index=False))
        if v["funnel_not_viable"]:
            print(f"  <-- MARCA: {', '.join(map(str, v['funnel_not_viable']))} con < 20 partidos tras el filtro de volumen. "
                  f"El horizonte {a.decision_h} no es viable ahí" + (": probar `--decision-h 6`." if a.decision_h == "24" else "."))
        for r in v["recommendation"]:
            print(f"\nRECOMENDACIÓN: {r}")
        g = v.pop("guardrail")
        hide = ("funnel", "recommendation", "horizons", "edge_dies_at_half_spread", "method_sensitive")
        print("\n" + json.dumps({k: x for k, x in v.items() if k not in hide}, indent=1, ensure_ascii=False))
        print("edge muere a medio spread:", v["edge_dies_at_half_spread"])
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
        df = with_book(pd.read_csv(a.data), "open", "shin", "6")
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
