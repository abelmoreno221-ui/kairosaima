import unittest
import numpy as np
from pmarb.devig import multiplicative, power, shin
from pmarb.model import Blend, brier
from pmarb.matching import match_event, outcome_of
from pmarb.backtest import Cfg, decide, run
from pmarb.dataset import synthetic


class T(unittest.TestCase):
    def test_devig_sums_to_one(self):
        for f in (multiplicative, power, shin):
            self.assertAlmostEqual(f([2.1, 3.4, 3.6]).sum(), 1, 6)

    def test_shin_shrinks_longshot(self):
        self.assertLess(shin([1.3, 5.5, 11])[2], multiplicative([1.3, 5.5, 11])[2])

    def test_blend_learns_sharper_source(self):
        df = synthetic(800)
        m = Blend().fit(df.p_pm, df.p_book, df.y)
        self.assertGreater(m.w[2], m.w[1])

    def test_matching(self):
        evs = [{"title": "Manchester United FC vs. Chelsea FC"}, {"title": "Arsenal FC vs. Everton FC"}]
        self.assertEqual(match_event("Man United", "Chelsea", evs)["title"], evs[0]["title"])
        self.assertIsNone(match_event("Leeds", "Burnley", evs))
        self.assertEqual(outcome_of("Will Chelsea FC win on 2025-03-16?", "Man United", "Chelsea"), "A")
        self.assertEqual(outcome_of("Will Man United vs. Chelsea end in a draw?", "Man United", "Chelsea"), "D")

    def test_decide_respects_costs(self):
        self.assertIsNone(decide(0.52, 0.50, Cfg(half_spread=0.01, min_edge=0.02)))
        self.assertEqual(decide(0.60, 0.50, Cfg())[0], "YES")
        self.assertEqual(decide(0.40, 0.50, Cfg())[0], "NO")

    def test_no_edge_when_sources_equal(self):
        # precios perfectos: cualquier "edge" es ruido de ajuste y no debe ser significativo
        s, _ = run(synthetic(800, pm_noise=0.0, book_noise=0.0), Cfg(half_spread=0.01))
        self.assertLessEqual(s["roi_per_trade_ci95"][0], 0)


if __name__ == "__main__":
    unittest.main()
