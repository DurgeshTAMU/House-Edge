from slot_forecasting.schema import PAYOUTS
from .features import HistoryIndex

class FeatureModel:
    def __init__(self, seed=20261107, augment=0):
        self.seed, self.augment = seed, augment

    def fit(self, queries, history, labels):
        import numpy as np
        from sklearn.ensemble import HistGradientBoostingClassifier
        index = HistoryIndex(history)
        examples, outcomes, weights = [], [], []
        for q in sorted(queries, key=lambda q:q.query_id):
            candidates = [(q.cutoff, labels[q.query_id])]
            eligible = [(t,y) for t,y in index.before(q.episode_id,q.machine_id,q.cutoff) if t >= 64]
            if self.augment and eligible:
                positions = np.linspace(0,len(eligible)-1,min(self.augment,len(eligible)),dtype=int)
                candidates += [eligible[i] for i in positions]
            for t,y in candidates:
                row = index.features(q,t)
                self.columns = sorted(row)
                examples.append([row[k] for k in self.columns])
                outcomes.append(y)
                weights.append(1/len(candidates))
        self.model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7,
            min_samples_leaf=30, l2_regularization=10, learning_rate=.05,
            early_stopping=False, random_state=self.seed)
        # Eight machines per episode; equal per-machine weights equalize episodes.
        weights = np.array(weights); weights *= len(weights)/weights.sum()
        self.model.fit(examples,outcomes,sample_weight=weights)
        self.training_rows = len(examples)
        return self

    def predict(self, queries, history):
        index = HistoryIndex(history)
        rows = [index.features(q) for q in queries]
        values = self.model.predict_proba([[r[k] for k in self.columns] for r in rows])
        result = {}
        for q, ps in zip(queries,values):
            mapped = dict(zip(self.model.classes_,ps))
            p = [float(mapped.get(y,0)) for y in PAYOUTS]
            result[q.query_id] = tuple(x/sum(p) for x in p)
        return result

class Blend:
    def __init__(self, a, b, weight):
        self.a,self.b,self.weight = a,b,weight
    def predict(self, queries, history):
        a,b = self.a.predict(queries,history),self.b.predict(queries,history)
        return {k:tuple(self.weight*x+(1-self.weight)*y for x,y in zip(a[k],b[k])) for k in a}
