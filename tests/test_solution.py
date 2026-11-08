import math
import pickle
import unittest
from dataclasses import replace
from slot_forecasting.schema import Query, Play, Target, CONDITIONS
from slot_forecasting.experiments import episode_folds
from slot_forecasting.baseline import FrequencyForecaster
from solution.features import HistoryIndex
from solution.models import FeatureModel, Blend

class FeaturesTest(unittest.TestCase):
    def setUp(self):
        self.q=Query('q','e','m','steady',0,10)
        self.h=[Play('e','m',1,0),Play('e','m',8,2),Play('e','other',9,5)]
    def test_future_invariance(self):
        original=HistoryIndex(self.h).features(self.q)
        extra=[Play(e,m,t,y) for e,m,t,y in [('e','m',10,20),('e','other',10,20),('e','m',11,1),('else','other',9,20)]]
        self.assertEqual(original,HistoryIndex(self.h+extra).features(self.q))
    def test_arbitrary_cutoff_matches_truncated_input(self):
        for t in (0,1,8,9,10):
            self.assertEqual(HistoryIndex(self.h).features(self.q,t),HistoryIndex([p for p in self.h if p.tick<t]).features(replace(self.q,cutoff=t)))
    def test_missing_is_not_zero_and_lags_are_ticks(self):
        f=HistoryIndex(self.h).features(self.q)
        self.assertEqual(f['lag1_missing'],1)
        self.assertEqual(f['lag1'],-1)
        self.assertEqual(f['lag2'],2)
        zero=HistoryIndex(self.h+[Play('e','m',9,0)]).features(self.q)
        self.assertEqual(zero['lag1'],0)
        self.assertEqual(zero['lag1_missing'],0)
        self.assertNotEqual(f['tick8_n'],zero['tick8_n'])
    def test_context_not_reused_historically(self):
        ix=HistoryIndex(self.h)
        self.assertEqual(ix.features(self.q,8),ix.features(replace(self.q,signal=1,machine_type='jackpot'),8))
    def test_empty_and_long_have_same_finite_schema(self):
        empty=HistoryIndex([]).features(self.q)
        full=HistoryIndex([Play('e','m',t,(0,1,2,5,20)[t%5]) for t in range(256)]).features(replace(self.q,cutoff=256))
        self.assertEqual(set(empty),set(full))
        self.assertTrue(all(math.isfinite(v) for v in [*empty.values(),*full.values()]))
    def test_duplicates_rejected(self):
        with self.assertRaises(ValueError): HistoryIndex(self.h+self.h)
    def test_episode_split(self):
        ts=[]; groups={}
        for c in CONDITIONS:
            for e in range(4):
                for m in range(2):
                    name=f'{c}.{e}.{m}'; ts.append(Target(name,f'{c}.{e}',(.2,)*5,0)); groups[name]=c
        a,b=episode_folds(ts,groups)
        self.assertFalse(a&b); self.assertEqual(a|b,{t.episode_id for t in ts})
        self.assertEqual((a,b),episode_folds(list(reversed(ts)),groups))

class ModelTest(unittest.TestCase):
    def test_class_alignment_and_checkpoint(self):
        class Dummy:
            classes_=[20,0]
            def predict_proba(self,x): return [[.3,.7] for _ in x]
        m=FeatureModel(); m.columns=sorted(HistoryIndex([]).features(Query('q','e','m','steady',0,10))); m.model=Dummy()
        q=[Query('q','e','m','steady',0,10)]
        self.assertEqual(m.predict(q,[])['q'],(.7,0,0,0,.3))
        base=FrequencyForecaster().fit(q,[],{'q':0})
        blend=Blend(base,base,.25)
        self.assertEqual(blend.predict(q,[]),pickle.loads(pickle.dumps(blend)).predict(q,[]))
        self.assertEqual(set(blend.predict(q,[])),{'q'})
        self.assertAlmostEqual(sum(blend.predict(q,[])['q']),1)

if __name__ == '__main__': unittest.main()
