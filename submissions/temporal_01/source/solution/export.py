"""Export a trusted local sklearn checkpoint; verify numerical parity before packaging."""
import argparse, hashlib, json, pickle, shutil, subprocess, sys, time, zipfile
from pathlib import Path
from threadpoolctl import threadpool_limits
from slot_forecasting.io import load_inputs, read_submission, write_submission, write_json
from slot_forecasting.submission_contract import package_code
from .portable import PortableModel

def export_model(model):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from .models import FeatureModel
    if type(model) is not FeatureModel or type(model.model) is not HistGradientBoostingClassifier:
        raise ValueError('Exporter supports only the tested history-feature HGB classifier')
    fitted=model.model
    if list(fitted.classes_) != [0,1,2,5,20]:
        raise ValueError('Unexpected class order')
    iterations=[]
    for predictors in fitted._predictors:
        if len(predictors) != 5: raise ValueError('Expected multiclass trees')
        trees=[]
        for predictor in predictors:
            nodes=[]
            for n in predictor.nodes:
                if n['is_categorical']: raise ValueError('Categorical splits are not supported')
                nodes.append([int(n['is_leaf']),float(n['value']),int(n['feature_idx']),
                    float(n['num_threshold']),int(n['missing_go_to_left']),int(n['left']),int(n['right'])])
            trees.append(nodes)
        iterations.append(trees)
    return {'format':'house-edge-numeric-hgb-v1','classes':[int(x) for x in fitted.classes_],
        'baseline':[float(v) for v in fitted._baseline_prediction[0]],
        'columns':model.columns,'iterations':iterations,
        'note':'Leaves already include learning-rate shrinkage; numeric <= thresholds.'}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True)
    p.add_argument('--data',type=Path,default=Path('public')); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists(): raise ValueError('Use a fresh export directory')
    a.out.mkdir(parents=True); source=a.out/'source'; source.mkdir()
    report=json.loads((a.run/'report.json').read_text(encoding='utf-8'))
    training_command=('python -m solution.cross_validate --config configs/train_cv_01.json --data public --out runs/new_run'
                      if 'cv_scores' in report else 'python -m solution.train --data public --out runs/new_run')
    model=pickle.loads((a.run/'model.pkl').read_bytes())
    artifact=export_model(model)
    write_json(source/'model.json',artifact)
    write_json(source/'training-information.json',{'chosen':report['chosen'],'seed':report.get('seed',20261107),
        'config':report.get('config'),'environment':report.get('environment'),'training_command':training_command})
    portable=PortableModel.load(source/'model.json')
    checks={}
    with threadpool_limits(limits=2):
        for split,filename in [('validation','validation_predictions.csv'),('test','submission.csv')]:
            q,h=load_inputs(a.data/split); start=time.perf_counter()
            prediction=portable.predict(q,h)
            expected=read_submission(a.run/filename,{v.query_id for v in q})
            delta=max(abs(x-y) for k in expected for x,y in zip(expected[k],prediction[k]))
            if delta>1e-8: raise ValueError('Portable parity failed')
            write_submission(a.out/filename,prediction)
            checks[split]={'queries':len(q),'max_probability_difference':delta,'seconds':time.perf_counter()-start}
    root=Path(__file__).resolve().parents[1]
    for folder,names in {'solution':['__init__.py','features.py','portable.py'],
                         'slot_forecasting':['__init__.py','io.py','schema.py']}.items():
        (source/folder).mkdir()
        for name in names: shutil.copyfile(root/folder/name,source/folder/name)
    (source/'predict.py').write_text('from solution.portable import main\nif __name__ == "__main__": main()\n',encoding='utf-8')
    write_json(source/'reproduce.json',{'image':'python','command':['python','predict.py','--data','/data','--out','/output/submission.csv']})
    (source/'README.md').write_text('# House Edge portable tree submission\n\n'
        'Python 3.11+ standard library only. No pip install, model retraining, pickle loading, network or APIs.\n'
        'Run from this folder: python predict.py --data /data --out /output/submission.csv\n'
        'model.json stores the frozen numeric trees and feature column order. solution/features.py reproduces training features.\n'
        'Exported from '+report['chosen']+' checkpoint; training settings and environment: training-information.json.\n'
        'Parity with the original model checked on every public validation/test query before packaging.\n'
        'Training reproduction in participant workspace: '+training_command+'\n'
        'Export command in participant workspace: python -m solution.export --run runs/new_run --data public --out submissions/new_portable\n'
        'Training code and dependencies belong to the participant workspace; this archive is inference-only.\n'
        'Upload submission.csv separately. Offline organizer paths: read-only /submission and /data; writable /output.\n',encoding='utf-8')
    archive=package_code(source,a.out/'entry.zip')
    extracted=a.out/'extracted'; extracted.mkdir()
    with zipfile.ZipFile(archive) as z: z.extractall(extracted)
    output=(a.out/'reproduced_windows.csv').resolve()
    subprocess.run([sys.executable,'-S','predict.py','--data',str(a.data.resolve()),'--out',str(output)],cwd=extracted,check=True)
    expected=read_submission(a.out/'submission.csv'); actual=read_submission(output,set(expected))
    delta=max(abs(x-y) for k in expected for x,y in zip(expected[k],actual[k]))
    if delta>1e-8: raise ValueError('Extracted package mismatch')
    checks['windows_without_site_packages']={'max_probability_difference':delta}
    checks['source_checkpoint_sha256']=hashlib.sha256((a.run/'model.pkl').read_bytes()).hexdigest()
    checks['zip_sha256']=hashlib.sha256(Path(archive).read_bytes()).hexdigest()
    checks['zip_bytes']=Path(archive).stat().st_size
    write_json(a.out/'verification.json',checks)
    print(json.dumps(checks,indent=2))
if __name__ == '__main__': main()
