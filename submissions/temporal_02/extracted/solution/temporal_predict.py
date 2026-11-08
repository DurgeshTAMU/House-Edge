"""Standard-library inference for a frozen temporal/boosted blend."""
import argparse,json
from pathlib import Path
from slot_forecasting.io import load_inputs,write_submission
from .portable import PortableModel
from .temporal import TemporalForecaster

def predict(queries,history,model_root):
    root=Path(model_root); config=json.loads((root/'blend.json').read_text())
    weight=config['temporal_weight']
    if not 0<=weight<=1: raise ValueError('Invalid blend weight')
    temporal=TemporalForecaster.load(root/'temporal.json').predict(queries,history)
    if weight==1: return temporal
    base=PortableModel.load(root/'model.json').predict(queries,history)
    return {k:tuple(weight*x+(1-weight)*y for x,y in zip(temporal[k],base[k])) for k in temporal}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model-root',type=Path,default=Path(__file__).resolve().parents[1]); a=p.parse_args()
    q,h=load_inputs(a.data/'test'); write_submission(a.out,predict(q,h,a.model_root))
if __name__=='__main__': main()
