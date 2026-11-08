import tempfile,unittest
from pathlib import Path
from slot_forecasting.schema import Query,Play
from solution.temporal import TemporalForecaster

class TemporalTest(unittest.TestCase):
    def model(self,**kwargs):
        m=TemporalForecaster(**kwargs); m.prior=[.5,.2,.15,.1,.05]; return m
    def test_missing_ticks_and_exact_lags(self):
        m=self.model()
        self.assertIsNone(m.context({1:0,8:2},10,(1,2)))
        self.assertEqual(m.context({8:2,9:0},10,(1,2)),(0,1))
    def test_future_outcomes_do_not_change_forecast(self):
        m=self.model(); rows=[(t,0 if t%2 else 2) for t in range(80)]
        self.assertEqual(m.machine(rows,60),m.machine([(t,y) for t,y in rows if t<60]+[(60,20),(70,1)],60))
    def test_chronological_alternation_learning(self):
        m=self.model(); rows=[(t,0 if t%2 else 2) for t in range(180)]
        p=m.machine(rows,180)
        self.assertGreater(p[2],.75)
        self.assertAlmostEqual(sum(p),1.)
    def test_short_empty_and_normalization(self):
        m=self.model()
        for rows,t in [([],0),([(1,20)],2),([(1,0),(40,2)],100)]:
            p=m.machine(rows,t)
            self.assertTrue(all(0<=v<=1 for v in p)); self.assertAlmostEqual(sum(p),1)
    def test_other_episodes_and_peer_rows_cannot_contaminate(self):
        m=self.model(); q=[Query('q','e','m','steady',0,10)]; h=[Play('e','m',8,0)]
        self.assertEqual(m.predict(q,h),m.predict(q,h+[Play('other','m',9,20),Play('e','peer',9,20)]))
    def test_json_roundtrip(self):
        m=self.model(categories=3); rows=[(t,(0,1,2,5,20)[t%5]) for t in range(60)]
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'model.json'; m.save(p)
            self.assertEqual(m.machine(rows,60),TemporalForecaster.load(p).machine(rows,60))

if __name__=='__main__': unittest.main()
