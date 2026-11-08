"""Package and verify a promoted standard-library temporal blend."""
import argparse,hashlib,json,shutil,subprocess,sys,time,zipfile
from pathlib import Path
from slot_forecasting.io import read_queries,read_submission,write_json
from slot_forecasting.submission_contract import package_code

def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True)
    p.add_argument('--data',type=Path,default=Path('public')); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    report=json.loads((a.run/'report.json').read_text())
    if not report['decision']['promoted']: raise ValueError('No promoted temporal model to package')
    if a.out.exists(): raise ValueError('Use a new package folder')
    source=a.out/'source'; source.mkdir(parents=True); root=Path(__file__).resolve().parents[1]
    blend=json.loads((a.run/'blend.json').read_text())
    names=['temporal.json','blend.json']+(['model.json'] if blend['temporal_weight']<1 else [])
    for name in names: shutil.copyfile(a.run/name,source/name)
    for folder,names in {'solution':['__init__.py','features.py','portable.py','temporal.py','temporal_predict.py'],
                         'slot_forecasting':['__init__.py','schema.py','io.py']}.items():
        (source/folder).mkdir()
        for name in names: shutil.copyfile(root/folder/name,source/folder/name)
    (source/'predict.py').write_text('from solution.temporal_predict import main\nif __name__ == "__main__": main()\n',encoding='utf-8')
    write_json(source/'reproduce.json',{'image':'python','command':['python','predict.py','--data','/data','--out','/output/submission.csv']})
    write_json(source/'training-information.json',{'chosen':report['chosen'],'config':report['config'],
        'input_sha256':report['input_sha256'],'environment':report['environment']})
    (source/'README.md').write_text('# House Edge temporal submission\n\n'
        'Python 3.11+ standard library only. No pip installation, network, pickle, API, or external file dependencies.\n'
        'Run: python predict.py --data /data --out /output/submission.csv\n'
        'Chosen model: '+report['chosen']+'. Training/configuration metadata: training-information.json.\n'
        'The temporal predictor updates machine-local counts using only supplied observations before cutoff.\n'
        'temporal.json stores global prior and settings; blend.json fixes the mixture weight. model.json is included only if the boosted component is used.\n'
        'Training from participant workspace: python -m solution.temporal_cv --config configs/temporal_cv_01.json --data public --out runs/new_run\n'
        'The incumbent model/OOF references in that configuration are required for training, not for this self-contained inference archive.\n'
        'Keep /submission and /data read-only; output is written to the requested /output path.\n'
        'Upload the matching forecast CSV separately.\n',encoding='utf-8')
    archive=package_code(source,a.out/'entry.zip'); extracted=a.out/'extracted'; extracted.mkdir()
    with zipfile.ZipFile(archive) as z:z.extractall(extracted)
    output=(a.out/'reproduced_windows.csv').resolve(); start=time.perf_counter()
    subprocess.run([sys.executable,'-B','-S','predict.py','--data',str(a.data.resolve()),'--out',str(output)],cwd=extracted,check=True)
    ids={q.query_id for q in read_queries(a.data/'test/queries.csv')}
    expected=read_submission(a.run/'submission.csv',ids); actual=read_submission(output,ids)
    delta=max(abs(x-y) for k in ids for x,y in zip(expected[k],actual[k]))
    if delta>1e-8:raise ValueError('Extracted prediction mismatch')
    shutil.copyfile(a.run/'submission.csv',a.out/'submission.csv')
    result={'queries':len(ids),'windows_max_probability_difference':delta,'seconds':time.perf_counter()-start,
        'zip_bytes':Path(archive).stat().st_size,'zip_sha256':hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
        'csv_sha256':hashlib.sha256((a.out/'submission.csv').read_bytes()).hexdigest()}
    write_json(a.out/'verification.json',result); print(json.dumps(result,indent=2))
if __name__=='__main__':main()
