import copy
import unittest
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from solution.models import FeatureModel
from solution.export import export_model
from solution.portable import PortableModel

class PortableTest(unittest.TestCase):
    def test_numeric_export_matches_sklearn_including_boundaries(self):
        rng=np.random.default_rng(51)
        x=rng.normal(size=(180,3)); x[::9,0]=np.nan
        y=np.array([0,1,2,5,20]*36)
        model=FeatureModel(); model.columns=['a','b','c']
        with threadpool_limits(limits=2):
            model.model=HistGradientBoostingClassifier(max_iter=8,max_leaf_nodes=4,min_samples_leaf=5,random_state=51).fit(x,y)
            artifact=export_model(model)
            # Exact <= boundaries, adjacent floats, and missing-value branches.
            probes=[list(row) for row in x[:20]]
            for tree in artifact['iterations'][0]:
                for node in tree:
                    if not node[0]:
                        for value in (node[3],np.nextafter(node[3],-np.inf),np.nextafter(node[3],np.inf),np.nan):
                            row=[0.,0.,0.]; row[node[2]]=value; probes.append(row)
            expected=model.model.predict_proba(probes)
            actual=PortableModel(artifact).predict_rows(probes)
            np.testing.assert_allclose(expected,actual,atol=1e-14,rtol=0)
        bad=copy.deepcopy(artifact); bad['classes'].reverse()
        with self.assertRaises(ValueError): PortableModel(bad)
        with self.assertRaises(ValueError): export_model(object())

    def test_softmax_stability(self):
        artifact={'format':'house-edge-numeric-hgb-v1','classes':[0,1,2,5,20],
                  'baseline':[10000.,-10000.,0.,0.,0.],'columns':[],'iterations':[]}
        p=PortableModel(artifact).predict_rows([[]])[0]
        self.assertEqual(p,(1.,0.,0.,0.,0.))

if __name__ == '__main__': unittest.main()
