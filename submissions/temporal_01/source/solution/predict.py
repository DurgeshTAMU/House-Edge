"""Load only a trusted locally produced checkpoint; no training/network access."""
import argparse
import pickle
from pathlib import Path
from threadpoolctl import threadpool_limits
from slot_forecasting.io import load_inputs, write_submission

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',type=Path,default=Path(__file__).resolve().parents[1]/'model.pkl')
    a=p.parse_args()
    with a.model.open('rb') as f: model=pickle.load(f)
    q,h=load_inputs(a.data/'test')
    with threadpool_limits(limits=2):
        write_submission(a.out,model.predict(q,h))
if __name__ == '__main__': main()
