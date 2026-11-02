import argparse
import json
import pandas as pd
from .backtest import Cfg, grid, table, with_book
from .model import Blend


def main():
    ap = argparse.ArgumentParser("pmarb")
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build", help="dataset histórico football-data x Polymarket")
    b.add_argument("--fd", nargs="+", required=True, help="CSV(s) de football-data.co.uk")
    b.add_argument("--tag", default="soccer")
    b.add_argument("--lead-min", type=int, default=10)
    b.add_argument("--out", default="dataset.csv")
    s = sp.add_parser("synthetic", help="dataset simulado para probar")
    s.add_argument("--out", default="dataset.csv")
    t = sp.add_parser("backtest")
    t.add_argument("data")
    t.add_argument("--fee-rate", type=float, default=0.05)
    t.add_argument("--min-edge", type=float, default=0.04)
    t.add_argument("--half-spread", type=float, default=0.01)
    t.add_argument("--min-liquidity", type=float, default=1000.0)
    t.add_argument("--json-out", default="backtest.json")
    f = sp.add_parser("fit", help="ajusta pesos finales para el escáner")
    f.add_argument("data")
    f.add_argument("--out", default="weights.json")
    c = sp.add_parser("scan", help="paper trading en vivo (requiere ODDS_API_KEY)")
    c.add_argument("--sport", default="soccer_epl")
    c.add_argument("--tag", default="soccer")
    a = ap.parse_args()

    if a.cmd == "build":
        from .dataset import build
        from .odds import load_football_data
        fd = pd.concat([load_football_data(p) for p in a.fd])
        build(fd, a.tag, a.lead_min).to_csv(a.out, index=False)
    elif a.cmd == "synthetic":
        from .dataset import synthetic
        synthetic().to_csv(a.out, index=False)
    elif a.cmd == "backtest":
        cfg = Cfg(fee_rate=a.fee_rate, min_edge=a.min_edge, half_spread=a.half_spread,
                  min_liquidity=a.min_liquidity)
        res, verdict = grid(pd.read_csv(a.data), cfg)
        import pandas as _pd
        with _pd.option_context("display.width", 250, "display.max_columns", 30):
            print(table(res).to_string(index=False))
        print(json.dumps(verdict, indent=2, ensure_ascii=False))
        seen = {}
        for (l, m), s in res.items():
            seen.setdefault(tuple(b["n_train"] for b in s["blocks"]), []).append(f"{l}/{m}")
        for sizes, who in seen.items():
            print(f"n_train por bloque ({', '.join(who)}):", " ".join(map(str, sizes)))
        json.dump({f"{l}/{m}": s for (l, m), s in res.items()} | {"verdict": verdict},
                  open(a.json_out, "w"), indent=2, default=str)
    elif a.cmd == "fit":
        df = with_book(pd.read_csv(a.data), "open", "shin")
        m = Blend().fit(df.p_pm.values, df.p_book.values, df.y.values)
        json.dump({"w": m.w.tolist(), "n": len(df)}, open(a.out, "w"))
        print("w =", m.w.round(3), "(intercepto, peso PM, peso casa)")
    elif a.cmd == "scan":
        from .scan import scan
        print(scan(a.sport, a.tag).to_string())


if __name__ == "__main__":
    main()
