"""Package trusted local model and verify it from a fresh extracted folder."""
import argparse, hashlib, json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path
from slot_forecasting.io import read_queries, read_submission, write_json
from slot_forecasting.submission_contract import package_code

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True); p.add_argument('--data',type=Path,default=Path('public'))
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); root=Path(__file__).resolve().parents[1]
    if a.out.exists(): raise ValueError('Choose a fresh packaging folder')
    source=a.out/'source'; source.mkdir(parents=True)
    for name in ('solution','slot_forecasting'):
        dest=source/name; dest.mkdir()
        for f in (root/name).glob('*.py'): shutil.copyfile(f,dest/f.name)
    shutil.copyfile(a.run/'model.pkl',source/'model.pkl')
    shutil.copyfile(root/'requirements-solution.txt',source/'requirements.txt')
    report=json.loads((a.run/'report.json').read_text(encoding='utf-8'))
    (source/'predict.py').write_text('from solution.predict import main\nif __name__ == "__main__": main()\n',encoding='utf-8')
    write_json(source/'reproduce.json',{'image':'python','command':['python','predict.py','--data','/data','--out','/output/submission.csv']})
    (source/'README.md').write_text('# House Edge CPU submission\n\n'
        'Install pinned requirements.txt before the offline run. Python 3.13 tested on Windows; Linux organizer runtime not tested.\n'
        'Run: python predict.py --data /data --out /output/submission.csv\n'
        'Uses a trusted included pickle checkpoint; no fitting, network or validation answers at inference.\n'
        'Chosen model: '+report['chosen']+'; seed 20261107; CPU thread limit 2.\n'
        'Training from participant repository: python -m solution.train --data public --out runs/new_run\n'
        'Training settings and versions: training-report.json. Upload forecast CSV separately.\n',encoding='utf-8')
    shutil.copyfile(a.run/'report.json',source/'training-report.json')
    archive=package_code(source,a.out/'entry.zip')
    fresh=a.out/'reproduction'; fresh.mkdir()
    with zipfile.ZipFile(archive) as z: z.extractall(fresh)
    output=(a.out/'reproduced.csv').resolve()
    env=os.environ.copy(); env.pop('PYTHONPATH',None)
    started=time.perf_counter()
    subprocess.run([sys.executable,'predict.py','--data',str(a.data.resolve()),'--out',str(output)],cwd=fresh,env=env,check=True)
    ids={q.query_id for q in read_queries(a.data/'test/queries.csv')}
    expected=read_submission(a.run/'submission.csv',ids); actual=read_submission(output,ids)
    delta=max(abs(x-y) for k in ids for x,y in zip(expected[k],actual[k]))
    if delta>1e-8: raise ValueError('Fresh-folder prediction mismatch')
    shutil.copyfile(a.run/'submission.csv',a.out/'submission.csv')
    result={'max_probability_difference':delta,'queries':len(ids),'seconds':time.perf_counter()-started,
            'zip_bytes':Path(archive).stat().st_size,'csv_bytes':output.stat().st_size,
            'same_environment_only':True,'zip_sha256':hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
            'csv_sha256':hashlib.sha256(output.read_bytes()).hexdigest()}
    write_json(a.out/'reproduction.json',result); print(json.dumps(result,indent=2))
if __name__ == '__main__': main()
