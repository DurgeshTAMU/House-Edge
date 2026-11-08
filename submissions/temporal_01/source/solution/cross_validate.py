"""Fixed, training-only model selection, then one descriptive validation evaluation."""
import argparse, hashlib, json, pickle, platform, time
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from slot_forecasting.schema import PAYOUTS
from slot_forecasting.io import load_inputs, read_labels, read_rows, write_json, write_submission
from slot_forecasting.rounds import round_metadata, score_round
from .models import FeatureModel
from .training_data import build_examples, validate_folds, fold_masks

def outcome_scores(prediction, outcomes, episodes, groups):
    by_episode=defaultdict(list)
    for p,y,e in zip(prediction,outcomes,episodes):
        errors=[(float(v)-float(y==k))**2 for v,k in zip(p,PAYOUTS)]
        by_episode[str(e)].append((sum(errors),errors[0]))
    by_condition=defaultdict(list)
    for e,values in by_episode.items(): by_condition[groups[e]].append(np.mean(values,axis=0))
    condition={c:{'brier':float(np.mean(v,axis=0)[0]),'zero_brier':float(np.mean(v,axis=0)[1])} for c,v in by_condition.items()}
    return {'brier':float(np.mean([v['brier'] for v in condition.values()])),
            'zero_brier':float(np.mean([v['zero_brier'] for v in condition.values()])), 'conditions':condition}

def choose_candidate(scores, config):
    best=min(scores,key=lambda n:(np.mean([s['brier'] for s in scores[n]]),n))
    base=float(np.mean([s['brier'] for s in scores['incumbent']]))
    value=float(np.mean([s['brier'] for s in scores[best]]))
    gain=(base-value)/base
    wins=sum(s['brier']<b['brier'] for s,b in zip(scores[best],scores['incumbent']))
    promote=best!='incumbent' and gain>=config['promotion_min_relative_gain'] and wins>=config['promotion_min_folds_improved']
    return {'best_cv':best,'selected':best if promote else 'incumbent','promoted':promote,
            'relative_gain':gain,'folds_improved':wins,'incumbent_brier':base,'best_brier':value}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,default=Path('public'))
    p.add_argument('--config',type=Path,default=Path('configs/train_cv_01.json')); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists(): raise ValueError('Choose a fresh run folder')
    a.out.mkdir(parents=True); start=time.perf_counter()
    config=json.loads(a.config.read_text()); write_json(a.out/'config.json',config)
    folds=json.loads(Path(config['fold_manifest']).read_text())['folds']; write_json(a.out/'folds.json',folds)
    q,h=load_inputs(a.data/'train'); labels=read_labels(a.data/'train/labels.csv',q)
    validate_folds(folds,{v.episode_id for v in q})
    condition_rows=read_rows(a.data/'train/conditions.csv',('query_id','condition'))
    by_id={r['query_id']:r['condition'] for r in condition_rows}
    groups={v.episode_id:by_id[v.query_id] for v in q}
    for v in q:
        if groups[v.episode_id]!=by_id[v.query_id]: raise ValueError('Mixed episode conditions')
    print('Building shared training features',flush=True)
    data=build_examples(q,h,labels,config['augment']); np.savez_compressed(a.out/'training_features.npz',**data)
    print('Feature rows:',len(data['y']),'columns:',len(data['columns']),flush=True)
    scores={n:[] for n in config['candidates']}; timings={n:[] for n in scores}
    oof={n:{} for n in scores}
    with threadpool_limits(limits=2):
        for fold,held in enumerate(folds):
            fit,evaluate=fold_masks(data,held)
            weight=data['weight'][fit].copy(); weight*=len(weight)/weight.sum()
            for name,params in config['candidates'].items():
                t=time.perf_counter()
                model=HistGradientBoostingClassifier(**params,early_stopping=False,random_state=config['seed'])
                model.fit(data['X'][fit],data['y'][fit],sample_weight=weight)
                if tuple(model.classes_)!=PAYOUTS: raise ValueError('Missing training class')
                predicted=model.predict_proba(data['X'][evaluate])
                result=outcome_scores(predicted,data['y'][evaluate],data['episode'][evaluate],groups)
                scores[name].append(result); timings[name].append(time.perf_counter()-t)
                oof[name].update({str(k):tuple(map(float,p)) for k,p in zip(data['query_id'][evaluate],predicted)})
                print('fold',fold,name,json.dumps(result),flush=True)
            write_json(a.out/'cv_progress.json',{'scores':scores,'seconds':timings})
        decision=choose_candidate(scores,config)
        # Persist decision before touching any validation/test file.
        write_json(a.out/'decision.json',decision)
        for name,predictions in oof.items(): write_submission(a.out/('oof_'+name+'.csv'),predictions)
        report={'chosen':decision['selected'],'decision':decision,'cv_scores':scores,'cv_seconds':timings,
                'config':config,'selection':'Training episodes only; validation is descriptive after selection',
                'seed':config['seed'],'training_rows':len(data['y'])}
        if decision['promoted']:
            selected=FeatureModel(seed=config['seed'],augment=config['augment']); selected.columns=list(data['columns'])
            selected.model=HistGradientBoostingClassifier(**config['candidates'][decision['selected']],early_stopping=False,random_state=config['seed'])
            weight=data['weight'].copy(); weight*=len(weight)/weight.sum()
            selected.model.fit(data['X'],data['y'],sample_weight=weight); selected.training_rows=len(data['y'])
            content=pickle.dumps(selected,protocol=5); (a.out/'model.pkl').write_bytes(content)
            restored=pickle.loads(content)
            vq,vh=load_inputs(a.data/'validation'); predicted=restored.predict(vq,vh)
            write_submission(a.out/'validation_predictions.csv',predicted)
            report['full_validation']=score_round(a.out/'validation_predictions.csv',a.data)
            tq,th=load_inputs(a.data/'test'); write_submission(a.out/'submission.csv',restored.predict(tq,th))
        else:
            report['retained_submission']='submissions/portable_01; no validation or test files read'
    import sklearn
    report['environment']={'python':platform.python_version(),'platform':platform.platform(),'numpy':np.__version__,'sklearn':sklearn.__version__,'cpu_thread_limit':2}
    report['seconds']=time.perf_counter()-start
    paths=[a.data/'train'/f for f in ('queries.csv','history.csv','labels.csv','conditions.csv')]
    report['input_sha256']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths}
    report['source_sha256']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in Path('solution').glob('*.py')}
    report['config_sha256']=hashlib.sha256(a.config.read_bytes()).hexdigest()
    write_json(a.out/'report.json',report)
    print(json.dumps({'decision':decision,'seconds':report['seconds'],'full_validation':report.get('full_validation')},indent=2),flush=True)
if __name__ == '__main__': main()
