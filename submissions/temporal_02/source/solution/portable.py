"""Standard-library inference for an explicitly exported numeric HGB model."""
import argparse
import json
import math
from pathlib import Path
from slot_forecasting.io import load_inputs, write_submission
from slot_forecasting.schema import PAYOUTS
from .features import HistoryIndex

class PortableModel:
    def __init__(self, artifact):
        if artifact.get('format') != 'house-edge-numeric-hgb-v1':
            raise ValueError('Unsupported portable model format')
        if artifact['classes'] != list(PAYOUTS):
            raise ValueError('Expected five payout classes in contract order')
        self.artifact = artifact

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding='utf-8')))

    def predict_rows(self, rows):
        out=[]
        for row in rows:
            raw=list(self.artifact['baseline'])
            for iteration in self.artifact['iterations']:
                for c,tree in enumerate(iteration):
                    node=tree[0]
                    while not node[0]:
                        # leaf, value, feature, threshold, missing-left, left, right
                        value=row[node[2]]
                        left=node[4] if math.isnan(value) else value <= node[3]
                        node=tree[node[5] if left else node[6]]
                    raw[c] += node[1]
            peak=max(raw)
            exp=[math.exp(v-peak) for v in raw]
            total=sum(exp)
            out.append(tuple(v/total for v in exp))
        return out

    def predict(self, queries, history):
        ix=HistoryIndex(history)
        columns=self.artifact['columns']
        features=[ix.features(q) for q in queries]
        rows=[[f[c] for c in columns] for f in features]
        return {q.query_id:p for q,p in zip(queries,self.predict_rows(rows))}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',type=Path,default=Path(__file__).resolve().parents[1]/'model.json')
    args=p.parse_args()
    model=PortableModel.load(args.model)
    q,h=load_inputs(args.data/'test')
    write_submission(args.out,model.predict(q,h))
if __name__ == '__main__': main()
