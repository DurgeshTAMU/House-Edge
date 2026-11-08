"""Bounded preregistered CPU comparison; select on tuning episodes only."""
import argparse, hashlib, json, pickle, platform, time
from pathlib import Path
from threadpoolctl import threadpool_limits
from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.transition import TransitionForecaster
from slot_forecasting.io import load_inputs, read_labels, read_targets, read_rows, read_submission, write_submission, write_json
from slot_forecasting.experiments import episode_folds, subset_score
from slot_forecasting.rounds import round_metadata, score_round
from .models import FeatureModel, Blend

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,default=Path('public')); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists(): raise ValueError('Use a fresh output folder')
    a.out.mkdir(parents=True)
    started=time.perf_counter(); meta=round_metadata(a.data)
    tq,th=load_inputs(a.data/'train'); labels=read_labels(a.data/'train/labels.csv',tq)
    vq,vh=load_inputs(a.data/'validation'); targets=read_targets(a.data/'validation/targets.csv')
    groups={r['query_id']:r['condition'] for r in read_rows(a.data/'validation/conditions.csv',('query_id','condition'))}
    ref=read_submission(a.data/'validation/reference.csv')
    tuning,audit=episode_folds(targets,groups)
    write_json(a.out/'folds.json',{'tuning':sorted(tuning),'audit':sorted(audit)})
    factories=[('pooled_recent',lambda:FrequencyForecaster('pooled_recent')),
               ('pooled',lambda:FrequencyForecaster('pooled')),
               ('features_queries',lambda:FeatureModel()),('features_augmented',lambda:FeatureModel(augment=12))]
    for lags in ((1,),(2,),(4,),(8,),(1,2),(2,4)):
        for half in (24,128):
            factories.append(('transition_'+ '_'.join(map(str,lags))+'_'+str(half),lambda lags=lags,half=half:TransitionForecaster(lags,half)))
    models={}; preds={}; comparison={}
    def record(name,model,pred,seconds):
        models[name]=model; preds[name]=pred
        comparison[name]={'tuning':subset_score(pred,targets,groups,ref,tuning,meta['max_points']),'fit_predict_seconds':seconds}
        print(name,comparison[name]['tuning']['technical_points'],flush=True)
    with threadpool_limits(limits=2):
        for name,factory in factories:
            start=time.perf_counter(); model=factory().fit(tq,th,labels)
            record(name,model,model.predict(vq,vh),time.perf_counter()-start)
        transition=min((n for n in models if n.startswith('transition')),key=lambda n:comparison[n]['tuning']['mean_loss_ratio'])
        for other in (transition,'features_queries','features_augmented'):
            for weight in (.25,.5,.75):
                name=f'blend_{other}_{weight}'
                pred={k:tuple(weight*x+(1-weight)*y for x,y in zip(preds['pooled_recent'][k],preds[other][k])) for k in preds[other]}
                record(name,Blend(models['pooled_recent'],models[other],weight),pred,0)
        chosen=min(models,key=lambda n:(comparison[n]['tuning']['mean_loss_ratio'],n))
        model=models[chosen]
        report={'chosen':chosen,'comparison':comparison,'round':meta,'seed':20261107,
                'selection':'Minimum tuning loss ratio; audit evaluated only after selection',
                'audit':subset_score(preds[chosen],targets,groups,ref,audit,meta['max_points']),
                'baseline_audit':subset_score(preds['pooled_recent'],targets,groups,ref,audit,meta['max_points'])}
        model_bytes=pickle.dumps(model,protocol=5); (a.out/'model.pkl').write_bytes(model_bytes)
        reloaded=pickle.loads(model_bytes); again=reloaded.predict(vq,vh)
        delta=max(abs(x-y) for k in again for x,y in zip(again[k],preds[chosen][k]))
        if delta > 1e-8: raise ValueError('Checkpoint reproduction mismatch')
        report['reload_max_difference']=delta
        write_submission(a.out/'validation_predictions.csv',preds[chosen])
        report['full_validation']=score_round(a.out/'validation_predictions.csv',a.data)
        xq,xh=load_inputs(a.data/'test')
        write_submission(a.out/'submission.csv',reloaded.predict(xq,xh))
    import numpy, sklearn, scipy, joblib, threadpoolctl
    versions={m.__name__:m.__version__ for m in (numpy,sklearn,scipy,joblib,threadpoolctl)}
    report['environment']={'python':platform.python_version(),'platform':platform.platform(),'device':'CPU','thread_limit':2,'versions':versions}
    report['training_rows']={n:models[n].training_rows for n in ('features_queries','features_augmented')}
    report['seconds']=time.perf_counter()-started
    paths=[a.data/'metadata.json']+[a.data/s/f for s,fs in {'train':['queries.csv','history.csv','labels.csv'],'validation':['queries.csv','history.csv','targets.csv','conditions.csv','reference.csv'],'test':['queries.csv','history.csv']}.items() for f in fs]
    report['input_sha256']={str(f.relative_to(a.data)):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths}
    report['source_sha256']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for folder in ('solution','slot_forecasting') for f in sorted(Path(folder).glob('*.py'))}
    write_json(a.out/'report.json',report)
    print(json.dumps({'chosen':chosen,'audit':report['audit'],'full_validation':report['full_validation'],'seconds':report['seconds']},indent=2))
if __name__ == '__main__': main()
