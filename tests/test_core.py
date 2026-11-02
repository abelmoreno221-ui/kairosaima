import json
import os
import tempfile
import unittest
from unittest import mock
import pandas as pd
from pmarb.devig import multiplicative, power, shin, METHODS
from pmarb.model import Blend
from pmarb.matching import match_event, outcome_of
from pmarb.backtest import (Cfg, decide, run, with_book, grid, fee_per_share, settle, half_spread_at,
                            check_single_source, _solo_fails)
from pmarb.odds import choose_book, load_football_data, TRACKS
from pmarb import spreads as spr
from pmarb.dataset import synthetic, coverage, LowCoverage
from pmarb import polymarket as pm


class Core(unittest.TestCase):
    def test_devig(self):
        for f in (multiplicative, power, shin):
            self.assertAlmostEqual(f([2.1, 3.4, 3.6]).sum(), 1, 6)
        self.assertLess(shin([1.3, 5.5, 11])[2], multiplicative([1.3, 5.5, 11])[2])
        self.assertEqual(set(METHODS), {"shin", "prop", "log"})

    def test_blend_learns_sharper_source(self):
        df = with_book(synthetic(800), "close", "shin")
        m = Blend().fit(df.p_pm, df.p_book, df.y)
        self.assertGreater(m.w[2], m.w[1])

    def test_matching(self):
        evs = [{"title": "Manchester United FC vs. Chelsea FC"}, {"title": "Arsenal FC vs. Everton FC"}]
        self.assertEqual(match_event("Man United", "Chelsea", evs)["title"], evs[0]["title"])
        self.assertIsNone(match_event("Leeds", "Burnley", evs))
        self.assertEqual(outcome_of("Will Chelsea FC win on 2025-03-16?", "Man United", "Chelsea"), "A")
        self.assertEqual(outcome_of("Will Man United vs. Chelsea end in a draw?", "Man United", "Chelsea"), "D")


class Costs(unittest.TestCase):
    def test_fee_formula(self):
        self.assertAlmostEqual(100 * fee_per_share(0.5, 0.05), 1.25)
        self.assertAlmostEqual(fee_per_share(0.5, 0.05) / 0.5, 0.025)

    def test_fee_charged_on_entry_win_or_lose(self):
        win, fw = settle(50.0, 0.5, 1, 0.05)
        lose, fl = settle(50.0, 0.5, 0, 0.05)
        self.assertAlmostEqual(fw, fl)                      # misma comisión gane o pierda
        self.assertAlmostEqual(fw, 100 * 0.05 * 0.25)       # 100 acciones a 0,50 -> 1,25 $
        self.assertAlmostEqual(lose, -50.0 - 1.25)          # la perdedora también la paga
        self.assertAlmostEqual(win, 100 - 50.0 - 1.25)

    def test_ev_on_ask_not_mid(self):
        cfg = Cfg(half_spread=0.03, fee_rate=0.0, min_edge=0.0)
        side, ask, ev, q = decide(0.56, 0.50, cfg)
        self.assertAlmostEqual(ask, 0.53)
        self.assertAlmostEqual(ev, (0.56 - 0.53) / 0.53)    # sobre ask; sobre mid sería 0,12
        self.assertIsNone(decide(0.54, 0.50, Cfg()))        # spread + fee se comen 4 pts

    def test_half_spread_is_half(self):
        _, ask, _, _ = decide(0.70, 0.50, Cfg(half_spread=0.02, min_edge=0))
        self.assertAlmostEqual(ask, 0.52)                   # mid + MEDIO spread, no + spread completo

    def test_spread_table(self):
        cfg = Cfg(spread_table=((0.0, 0.2, 0, 48, 0.03), (0.2, 1.0, 0, 6, 0.01), (0.2, 1.0, 6, 48, 0.02)))
        self.assertEqual(half_spread_at(0.1, cfg, 24), 0.03)
        self.assertEqual(half_spread_at(0.5, cfg, 1), 0.01)
        self.assertEqual(half_spread_at(0.5, cfg, 24), 0.02)
        self.assertEqual(half_spread_at(0.5, cfg, 100), 0.03)   # celda vacía -> el peor


def _fake_trades(newest, n, step=60):
    """/trades simulado: más reciente primero."""
    data = [{"timestamp": newest - i * step, "size": 10, "price": 0.5} for i in range(n)]
    return lambda url, params: data[params["offset"]:params["offset"] + params["limit"]]


class Liquidity(unittest.TestCase):
    def _patch(self, fake, d):
        return (mock.patch.object(pm, "CACHE_DIR", d), mock.patch.object(pm, "_MIN_INTERVAL", 0),
                mock.patch.object(pm, "get_json", side_effect=fake))

    def test_paginates_until_before_T(self):
        fake = _fake_trades(newest=100_000, n=3000)
        with tempfile.TemporaryDirectory() as d:
            p1, p2, p3 = self._patch(fake, d)
            with p1, p2, p3 as g:
                c = pm.trades("0xa", until_ts=100_000 - 700 * 60, page=500)
                self.assertEqual(g.call_count, 2)            # para en cuanto pasa de T
                self.assertFalse(c["capped"])
                self.assertIsNotNone(pm.volume_window(c, 100_000 - 100 * 60, 300 * 60))
                pm.trades("0xa", until_ts=100_000 - 700 * 60, page=500)
                self.assertEqual(g.call_count, 2)            # caché: una vez por mercado

    def test_capped_offset_is_unknown_not_zero(self):
        fake = _fake_trades(newest=10**9, n=50_000)          # mercado muy líquido
        with tempfile.TemporaryDirectory() as d:
            p1, p2, p3 = self._patch(fake, d)
            with p1, p2, p3:
                c = pm.trades("0xb", until_ts=10**9 - 40_000 * 60, page=500, max_offset=10_000)
                self.assertTrue(c["capped"])
                self.assertIsNone(pm.volume_window(c, 10**9 - 30_000 * 60, 3600))

    def test_volume_window_strictly_before(self):
        c = {"trades": [{"t": 100, "usd": 10.0}, {"t": 200, "usd": 5.0}], "oldest": 100,
             "exhausted": True, "capped": False}
        self.assertEqual(pm.volume_window(c, 200, 1000), 10.0)

    def test_unknown_volume_reported_apart(self):
        df = synthetic(600)
        df.loc[df.match_id % 3 == 0, ["vol_open", "vol_close"]] = float("nan")
        s, t = run(with_book(df, "close", "shin"), Cfg(min_volume=0))
        u = s["unknown_volume"]
        self.assertGreater(u["markets"], 0)
        self.assertGreater(u["signals"], 0)
        self.assertEqual(s["signals"], len(t))               # cifras principales sin desconocidos

    def test_min_volume_filters(self):
        df = with_book(synthetic(600), "close", "shin")
        self.assertGreater(run(df, Cfg(min_volume=0))[0]["signals"], 0)
        self.assertEqual(run(df, Cfg(min_volume=1e12))[0]["signals"], 0)


class Sources(unittest.TestCase):
    def _csv(self, d, name, books):
        cols = {"Div": ["E0"] * 2, "Date": ["01/09/2025", "10/01/2026"], "HomeTeam": ["A", "C"],
                "AwayTeam": ["B", "D"], "FTR": ["H", "A"]}
        for b in books:
            for suf, v in (("H", 2.0), ("D", 3.4), ("A", 3.8), ("CH", 2.1), ("CD", 3.3), ("CA", 3.7)):
                cols[f"{b}{suf}"] = [v, v]
        path = os.path.join(d, name)
        pd.DataFrame(cols).to_csv(path, index=False)
        return path

    def test_fallback_avg_before_max(self):
        raws = [pd.DataFrame({"AvgH": [2.0], "MaxH": [2.2]})]
        self.assertEqual(choose_book(raws, "PS", log=lambda *_: None), "Avg")
        self.assertEqual(choose_book([pd.DataFrame({"MaxH": [2.2]})], "PS", log=lambda *_: None), "Max")

    def test_one_source_for_all_files(self):
        with tempfile.TemporaryDirectory() as d:
            f1, f2 = self._csv(d, "a.csv", ["PS", "Avg"]), self._csv(d, "b.csv", ["Avg"])
            df = load_football_data([f1, f2], "PS", log=lambda *_: None)
            self.assertEqual(set(df.source), {"Avg"})        # no PS en un fichero y Avg en otro

    def test_track_A_cutoff(self):
        with tempfile.TemporaryDirectory() as d:
            df = load_football_data(self._csv(d, "a.csv", ["PS"]), "PS", TRACKS["A"]["until"], log=lambda *_: None)
            self.assertEqual(len(df), 1)                     # el partido del 10/01/2026 queda fuera
        self.assertEqual(TRACKS["B"]["book"], "BFE")

    def test_mixed_sources_abort(self):
        df = synthetic(50)
        df.loc[df.match_id < 25, "source"] = "PS"
        df.loc[df.match_id >= 25, "source"] = "BFE"
        with self.assertRaises(SystemExit):
            check_single_source(df)


class Spreads(unittest.TestCase):
    def test_summary_cut_by_price_and_time(self):
        rows = [{"mid": 0.5, "hours_to_kickoff": h, "half_spread": hs}
                for h, hs in ((2, 0.01), (30, 0.03)) for _ in range(6)]
        rows += [{"mid": 0.5, "hours_to_kickoff": 60, "half_spread": 0.5}] * 6   # >48h: fuera
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.csv")
            pd.DataFrame(rows).to_csv(p, index=False)
            _, table = spr.summary(p, os.path.join(d, "t.json"))
        cfg = Cfg(spread_table=tuple(map(tuple, table)))
        self.assertAlmostEqual(half_spread_at(0.5, cfg, 2), 0.01)
        self.assertAlmostEqual(half_spread_at(0.5, cfg, 30), 0.03)
        self.assertLessEqual(max(x[4] for x in table), 0.03)


class Coverage(unittest.TestCase):
    def test_aborts_on_low_overlap(self):
        fd = pd.DataFrame({"date": pd.to_datetime(["2025-09-01"] * 3).date, "league": "E0",
                           "HomeTeam": ["Arsenal", "Leeds", "Wolves"], "AwayTeam": ["Chelsea", "Burnley", "Fulham"]})
        evs = [{"title": "Arsenal FC vs. Chelsea FC", "startDate": "2025-09-01"}]
        with self.assertRaises(LowCoverage):
            coverage(fd, evs, min_rate=0.5, min_matches=1, log=lambda *_: None)
        self.assertEqual(len(coverage(fd, evs, min_rate=0.3, min_matches=1, log=lambda *_: None)), 1)


class Grid(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res, cls.v = grid(synthetic(400), Cfg(min_train=200), spreads=(0.01, 0.05))

    def test_grid_shape_and_reports(self):
        self.assertEqual({k[0] for k in self.res}, {"open", "close"})
        self.assertEqual({k[2] for k in self.res}, {"0.01", "0.05"})
        s = self.res[("open", "shin", "0.01")]
        self.assertTrue(all("n_train" in b for b in s["blocks"]))
        self.assertEqual(set(s["groups"]), {"solo_modelo", "con_pinnacle"})
        self.assertEqual(s["signals"], sum(g["n"] for g in s["groups"].values()))
        for k in ("edge_dies_at_half_spread", "guardrail", "guardrail_warning", "verdict", "horizons"):
            self.assertIn(k, self.v)
        self.assertIn("referencia", self.v["verdict"])      # fuente SIM != Pinnacle -> no es veredicto

    def test_solo_modelo_losing_downgrades(self):
        s = {"groups": {"solo_modelo": {"n": 40, "roi": -0.05}}}
        self.assertTrue(_solo_fails(s))
        s["groups"]["solo_modelo"]["n"] = 10
        self.assertFalse(_solo_fails(s))

    def test_verdict_no_candidata(self):
        import pmarb.backtest as bt
        with mock.patch.object(bt, "_solo_fails", return_value=True), mock.patch.object(bt, "_alive", return_value=True):
            _, v = grid(synthetic(400).assign(source="PS"), Cfg(min_train=200), methods=("shin",), spreads=(0.01,))
        self.assertTrue(v["verdict"].startswith("no candidata"))

    def test_wider_spread_fewer_signals(self):
        self.assertGreaterEqual(self.res[("open", "shin", "0.01")]["signals"],
                                self.res[("open", "shin", "0.05")]["signals"])

    def test_no_edge_when_sources_equal(self):
        df = with_book(synthetic(800, pm_noise=0.0, close_noise=0.0, margin=0.0), "close", "prop")
        df["p_pm"] = df.p_book
        s, _ = run(df, Cfg(fee_rate=0.0, min_edge=0.02))
        self.assertLessEqual((s["roi_ci95"] or [0])[0], 0)


if __name__ == "__main__":
    unittest.main()
