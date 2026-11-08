"""Machine-specific lag-pair mixture with chronological evidence and shrinkage."""
import math
from itertools import combinations
from pathlib import Path
import json
from slot_forecasting.schema import PAYOUTS, probabilities, index_history
from .features import HistoryIndex

class TemporalForecaster:
    def __init__(self, half_life=128, score_half_life=128, categories=2, strength=8, size_strength=8):
        if min(half_life,score_half_life,strength,size_strength)<=0 or categories not in (2,3):
            raise ValueError('Invalid temporal model settings')
        self.settings=dict(half_life=half_life,score_half_life=score_half_life,categories=categories,strength=strength,size_strength=size_strength)
        self.prior=None
        self.lags=[(lag,) for lag in range(1,9)]+list(combinations(range(1,9),2))

    def fit(self, queries, history, labels):
        # No query context is assumed historically available.
        indexed=index_history(queries,history)
        counts=[.5]*5
        for q in sorted(queries,key=lambda q:q.query_id):
            for p in indexed.get((q.episode_id,q.machine_id),[]): counts[PAYOUTS.index(p.payout)]+=1
            counts[PAYOUTS.index(labels[q.query_id])]+=1
        self.prior=[v/sum(counts) for v in counts]
        return self

    def context(self, seq, tick, lags):
        values=[seq.get(tick-lag) for lag in lags]
        if any(v is None for v in values): return None
        if self.settings['categories']==2: return tuple(int(v>0) for v in values)
        return tuple(0 if v==0 else 1 if v==1 else 2 for v in values)

    def machine(self, rows, cutoff):
        if self.prior is None: raise ValueError('Fit or load before inference')
        # Filter again so arbitrary earlier-cutoff calls cannot use future records.
        rows=sorted((t,y) for t,y in rows if t<cutoff)
        seq={}; root=[0.]*5; tables=[{} for _ in self.lags]; score=[0.]*len(self.lags)
        previous=0
        half=self.settings['half_life']; score_half=self.settings['score_half_life']
        strength=self.settings['strength']; size_strength=self.settings['size_strength']
        def context_counts(table,key,t):
            if key is None: return [0.]*5
            if key not in table: table[key]=([0.]*5,t)
            values,last=table[key]
            decay=2**(-(t-last)/half)
            values=[v*decay for v in values]; table[key]=(values,t)
            return values
        def predict_at(t):
            total=sum(root)
            base=[(v+20*p)/(total+20) for v,p in zip(root,self.prior)]
            keys=[self.context(seq,t,lags) for lags in self.lags]
            forecasts=[]
            for table,key in zip(tables,keys):
                counts=context_counts(table,key,t); n=sum(counts)
                zero=(counts[0]+strength*base[0])/(n+strength)
                positive=sum(counts[1:]); prior_positive=sum(base[1:])
                sizes=[(c+size_strength*p/prior_positive)/(positive+size_strength) for c,p in zip(counts[1:],base[1:])]
                forecasts.append([zero]+[(1-zero)*p for p in sizes])
            return base,keys,forecasts
        for t,y in rows:
            decay=2**(-(t-previous)/half); score_decay=2**(-(t-previous)/score_half)
            root=[v*decay for v in root]; score=[v*score_decay for v in score]
            base,keys,forecasts=predict_at(t)
            target=PAYOUTS.index(y)
            for i,(table,key,pred) in enumerate(zip(tables,keys,forecasts)):
                # Score forecast BEFORE revealing y, then update sufficient statistics.
                if key is not None:
                    score[i]+=math.log(max(pred[target],1e-15))-math.log(max(base[target],1e-15))
                    table[key][0][target]+=1
            root[target]+=1; seq[t]=y; previous=t
        decay=2**(-(cutoff-previous)/half); score_decay=2**(-(cutoff-previous)/score_half)
        root=[v*decay for v in root]; score=[v*score_decay for v in score]
        base,keys,forecasts=predict_at(cutoff)
        # Half prior mass on context-free model, quarter each on single/pair families.
        logits=[math.log(.5)]+[s+math.log(.25/(8 if len(lags)==1 else 28)) for s,lags in zip(score,self.lags)]
        top=max(logits); weights=[math.exp(v-top) for v in logits]; total=sum(weights)
        result=[sum(w*p[k] for w,p in zip(weights,[base]+forecasts))/total for k in range(5)]
        return probabilities(result)

    def predict(self, queries, history):
        index=HistoryIndex(history)
        return {q.query_id:self.machine(index.before(q.episode_id,q.machine_id,q.cutoff),q.cutoff) for q in queries}

    def save(self,path):
        value={'format':'house-edge-temporal-v1','settings':self.settings,'prior':self.prior}
        Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')

    @classmethod
    def load(cls,path):
        value=json.loads(Path(path).read_text(encoding='utf-8'))
        if value.get('format')!='house-edge-temporal-v1': raise ValueError('Unknown model format')
        model=cls(**value['settings']); model.prior=list(probabilities(value['prior'])); return model
