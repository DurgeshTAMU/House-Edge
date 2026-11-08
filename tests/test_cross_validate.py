import unittest
import numpy as np
from slot_forecasting.schema import Query, Play
from solution.training_data import build_examples, fold_masks, validate_folds
from solution.cross_validate import choose_candidate, outcome_scores

class CrossValidationTest(unittest.TestCase):
    def test_all_historical_rows_follow_episode(self):
        q=[Query('q'+str(i),'e'+str(i),'m','steady',0,80) for i in range(3)]
        h=[Play(v.episode_id,'m',t,(0,1,2,5,20)[t%5]) for v in q for t in range(80)]
        data=build_examples(q,h,{v.query_id:0 for v in q},3)
        validate_folds([['e0'],['e1'],['e2']],data['episode'])
        fit,evaluate=fold_masks(data,['e1'])
        self.assertNotIn('e1',data['episode'][fit]); self.assertEqual(set(data['episode'][evaluate]),{'e1'})
        self.assertTrue(data['terminal'][evaluate].all())
        self.assertEqual(int(evaluate.sum()),1)
        for e in ('e0','e1','e2'): self.assertAlmostEqual(data['weight'][data['episode']==e].sum(),1)
        # Augmented target at t is never included in its own historical features.
        for row,t in zip(data['X'],data['cutoff']):
            self.assertEqual(row[list(data['columns']).index('all_n')],t)
    def test_bad_fold_manifests_rejected(self):
        for folds in ([['a'],['a']], [['a']], [['a'],[]]):
            with self.assertRaises(ValueError): validate_folds(folds,{'a','b'})
    def test_promotion_requires_size_and_consistency(self):
        config={'promotion_min_relative_gain':.005,'promotion_min_folds_improved':2}
        base=[{'brier':1.}]*3
        tiny=[{'brier':.999}]*3
        self.assertFalse(choose_candidate({'incumbent':base,'other':tiny},config)['promoted'])
        one=[{'brier':.8},{'brier':1.01},{'brier':1.01}]
        self.assertFalse(choose_candidate({'incumbent':base,'other':one},config)['promoted'])
        good=[{'brier':.98},{'brier':.98},{'brier':1.}]
        self.assertTrue(choose_candidate({'incumbent':base,'other':good},config)['promoted'])
    def test_score_weights_episodes_then_conditions(self):
        p=np.array([[1,0,0,0,0],[0,1,0,0,0],[0,1,0,0,0]])
        result=outcome_scores(p,[0,0,0],['a','b','b'],{'a':'stable','b':'drift'})
        self.assertEqual(result['brier'],1.)
        self.assertEqual(result['zero_brier'],.5)

if __name__ == '__main__': unittest.main()
