import json
import os
import tempfile
import unittest
from unittest import mock
import pandas as pd
from pmarb.devig import multiplicative, power, shin, METHODS
from pmarb.model import Blend
from pmarb.matching import match_event, outcome_of
from pmarb.backtest import Cfg, decide, run, with_book, grid, fee_per_share, settle, half_spread_at
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
        cfg = Cfg(spread_table=((0.0, 0.2, 0.03), (0.2, 1.0, 0.01)))
        self.assertEqual(half_spread_at(0.1, cfg), 0.03)
        self.assertEqual(half_spread_at(0.5, cfg), 0.01)


class Liquidity(unittest.TestCase):
    def test_volume_strictly_before(self):
        tl = [{"t": 100, "usd": 10.0}, {"t": 200, "usd": 5.0}, {"t": 300, "usd": 7.0}]
        self.assertEqual(pm.volume_before(tl, 200), 10.0)

    def test_trades_cached_one_call_per_market(self):
        with tempfile.TemporaryDirectory() as d, \
             mock.patch.object(pm, "CACHE_DIR", d), mock.patch.object(pm, "_MIN_INTERVAL", 0), \
             mock.patch.object(pm, "get_json", return_value=[{"timestamp": 1, "size": 10, "price": 0.5}]) as g:
            a = pm.trades("0xabc")
            b = pm.trades("0xabc")
            self.assertEqual(a, b)
            self.assertEqual(g.call_count, 1)

    def test_min_volume_filters(self):
        df = synthetic(600)
        s_all, _ = run(with_book(df, "close", "shin"), Cfg(min_volume=0))
        s_flt, _ = run(with_book(df, "close", "shin"), Cfg(min_volume=1e12))
        self.assertGreater(s_all["signals"], 0)
        self.assertEqual(s_flt["signals"], 0)


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
        for k in ("edge_dies_at_half_spread", "guardrail", "guardrail_warning", "verdict"):
            self.assertIn(k, self.v)

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
