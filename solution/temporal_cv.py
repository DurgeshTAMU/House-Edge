"""Evaluate fixed temporal candidates against cached, matching incumbent OOF forecasts."""
import argparse,hashlib,json,platform,shutil,time
from pathlib import Path
from slot_forecasting.io import load_inputs,read_labels,read_rows,read_submission,write_submission,write_json
from slot_forecasting.rounds import score_round
from .cross_validate import outcome_scores,choose_candidate
from .training_data import validate_folds
from .temporal import TemporalForecaster
from .temporal_predict import predict as predict_blend

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,default=Path('public'))
    p.add_argument('--config',type=Path,default=Path('configs/temporal_cv_01.json')); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists(): raise ValueError('Choose a fresh experiment folder')
    a.out.mkdir(parents=True); start=time.perf_counter()
    config=json.loads(a.config.read_text()); write_json(a.out/'config.json',config)
    prior_report=json.loads(Path(config['incumbent_report']).read_text())
    input_hashes={}
    for name in ('queries.csv','history.csv','labels.csv','conditions.csv'):
        path=a.data/'train'/name; digest=hashlib.sha256(path.read_bytes()).hexdigest()
        old=[value for key,value in prior_report['input_sha256'].items() if Path(key).name==name]
        if old!=[digest]: raise ValueError('Incumbent OOF data provenance mismatch')
        input_hashes[str(path)]=digest
    q,h=load_inputs(a.data/'train'); labels=read_labels(a.data/'train/labels.csv',q)
    folds=json.loads(Path(config['fold_manifest']).read_text()); validate_folds(folds,{v.episode_id for v in q})
    if folds!=json.loads((Path(config['incumbent_report']).parent/'folds.json').read_text()): raise ValueError('Incumbent folds differ')
    write_json(a.out/'folds.json',folds)
    base=read_submission(config['incumbent_oof'],{v.query_id for v in q})
    by_id={r['query_id']:r['condition'] for r in read_rows(a.data/'train/conditions.csv',('query_id','condition'))}
    groups={v.episode_id:by_id[v.query_id] for v in q}
    specs={}
    for name,params in config['candidates'].items():
        for weight in [1.]+config['blend_temporal_weights']:
            specs[name+'_w'+str(weight)]=(name,weight)
    scores={name:[] for name in ['incumbent',*specs]}; oof={name:{} for name in specs}; times={}
    for fold,held in enumerate(folds):
        held=set(held); tq=[v for v in q if v.episode_id not in held]; th=[v for v in h if v.episode_id not in held]
        vq=[v for v in q if v.episode_id in held]; vh=[v for v in h if v.episode_id in held]
        def score(pred):
            return outcome_scores([pred[v.query_id] for v in vq],[labels[v.query_id] for v in vq],[v.episode_id for v in vq],groups)
        scores['incumbent'].append(score(base))
        for variant,params in config['candidates'].items():
            tick=time.perf_counter()
            model=TemporalForecaster(**params).fit(tq,th,{v.query_id:labels[v.query_id] for v in tq})
            temporal=model.predict(vq,vh)
            times[str(fold)+'_'+variant]=time.perf_counter()-tick
            for name,(kind,weight) in specs.items():
                if kind!=variant: continue
                pred={k:tuple(weight*x+(1-weight)*y for x,y in zip(temporal[k],base[k])) for k in temporal}
                oof[name].update(pred); result=score(pred); scores[name].append(result)
                print('fold',fold,name,'brier',round(result['brier'],6),'zero',round(result['zero_brier'],6),flush=True)
            write_json(a.out/'progress.json',{'scores':scores,'seconds':times})
    decision=choose_candidate(scores,config)
    write_json(a.out/'decision.json',decision)
    for name,pred in oof.items(): write_submission(a.out/('oof_'+name+'.csv'),pred)
    report={'decision':decision,'chosen':decision['selected'],'cv_scores':scores,'timings':times,'config':config,
        'selection':'Training-only OOF comparison; previously used folds are development data, not fresh independent confirmation',
        'input_sha256':input_hashes,'incumbent_oof_sha256':hashlib.sha256(Path(config['incumbent_oof']).read_bytes()).hexdigest(),
        'environment':{'python':platform.python_version(),'platform':platform.platform(),'device':'CPU single-threaded temporal inference'}}
    if decision['promoted']:
        variant,weight=specs[decision['selected']]
        model=TemporalForecaster(**config['candidates'][variant]).fit(q,h,labels); model.save(a.out/'temporal.json')
        shutil.copyfile(config['incumbent_model'],a.out/'model.json')
        write_json(a.out/'blend.json',{'temporal_weight':weight,'variant':variant})
        vq,vh=load_inputs(a.data/'validation'); write_submission(a.out/'validation_predictions.csv',predict_blend(vq,vh,a.out))
        report['full_validation']=score_round(a.out/'validation_predictions.csv',a.data)
        xq,xh=load_inputs(a.data/'test'); write_submission(a.out/'submission.csv',predict_blend(xq,xh,a.out))
    else:
        report['retained_submission']='submissions/portable_01; validation/test not read'
    report['seconds']=time.perf_counter()-start
    report['source_sha256']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in Path('solution').glob('*.py')}
    report['config_sha256']=hashlib.sha256(a.config.read_bytes()).hexdigest()
    write_json(a.out/'report.json',report)
    print(json.dumps({'decision':decision,'seconds':report['seconds'],'full_validation':report.get('full_validation')},indent=2),flush=True)
if __name__=='__main__': main()
