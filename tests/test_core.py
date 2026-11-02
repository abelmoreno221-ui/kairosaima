import unittest
import numpy as np
from pmarb.devig import multiplicative, power, shin, METHODS
from pmarb.model import Blend, brier
from pmarb.matching import match_event, outcome_of
from pmarb.backtest import Cfg, decide, run, with_book, grid, fee_per_share
from pmarb.dataset import synthetic


class T(unittest.TestCase):
    def test_devig_sums_to_one(self):
        for f in (multiplicative, power, shin):
            self.assertAlmostEqual(f([2.1, 3.4, 3.6]).sum(), 1, 6)

    def test_shin_shrinks_longshot(self):
        self.assertLess(shin([1.3, 5.5, 11])[2], multiplicative([1.3, 5.5, 11])[2])

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

    def test_decide_respects_costs(self):
        # 4 pts de prob. a 0,50: tras spread (1) y fee (2,5 % del stake) queda < 4 % neto
        self.assertIsNone(decide(0.54, 0.50, Cfg()))
        self.assertEqual(decide(0.60, 0.50, Cfg())[0], "YES")
        self.assertEqual(decide(0.40, 0.50, Cfg())[0], "NO")

    def test_fee_formula(self):
        self.assertAlmostEqual(100 * fee_per_share(0.5, 0.05), 1.25)        # tope 1,25 $ / 100 acciones
        self.assertAlmostEqual(fee_per_share(0.5, 0.05) / 0.5, 0.025)       # 2,5 % del stake
        self.assertLess(fee_per_share(0.9, 0.05), fee_per_share(0.5, 0.05))

    def test_three_devig_methods(self):
        self.assertEqual(set(METHODS), {"shin", "prop", "log"})

    def test_grid_reports_open_and_close(self):
        res, v = grid(synthetic(400), Cfg(min_train=200))
        self.assertEqual({l for l, _ in res}, {"open", "close"})
        self.assertIn("verdict", v)
        s = res[("open", "shin")]
        self.assertTrue(all("n_train" in b for b in s["blocks"]))
        self.assertIn("mean", s["signals_per_week"])

    def test_no_edge_when_sources_equal(self):
        # precios perfectos: cualquier "edge" es ruido de ajuste y no debe ser significativo
        df = with_book(synthetic(800, pm_noise=0.0, close_noise=0.0, margin=0.0), "close", "prop")
        df["p_pm"] = df.p_book
        s, _ = run(df, Cfg(fee_rate=0.0, min_edge=0.02))
        self.assertLessEqual(s.get("roi_ci95", [0])[0], 0)


if __name__ == "__main__":
    unittest.main()
