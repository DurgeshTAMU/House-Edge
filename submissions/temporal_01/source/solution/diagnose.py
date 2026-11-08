"""Inspect tuning episodes only; do not reopen the reserved audit for model tuning."""
import argparse, csv, hashlib, json, random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from slot_forecasting.io import read_queries, read_targets, read_submission, read_rows, write_json
from slot_forecasting.schema import CONDITIONS

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,default=Path('public'))
    p.add_argument('--run',type=Path,default=Path('runs/cpu_comparison_01'))
    p.add_argument('--baseline',type=Path,default=Path('runs/baseline_pooled_recent'))
    p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    if a.out.exists(): raise ValueError('Choose a new diagnostic directory')
    a.out.mkdir(parents=True)
    folds=json.loads((a.run/'folds.json').read_text()); allowed=set(folds['tuning'])
    queries={q.query_id:q for q in read_queries(a.data/'validation/queries.csv') if q.episode_id in allowed}
    targets=[t for t in read_targets(a.data/'validation/targets.csv') if t.episode_id in allowed]
    selected=read_submission(a.run/'validation_predictions.csv')
    baseline=read_submission(a.baseline/'validation_predictions.csv')
    reference=read_submission(a.data/'validation/reference.csv')
    groups={r['query_id']:r['condition'] for r in read_rows(a.data/'validation/conditions.csv',('query_id','condition'))}
    rows=[]; episodes=defaultdict(list); payouts=defaultdict(list); types=defaultdict(list)
    for t in targets:
        q=queries[t.query_id]
        errors=[[sum((p-y)**2 for p,y in zip(pred[t.query_id],t.probabilities)) for pred in (selected,baseline,reference)]][0]
        row={'episode_id':q.episode_id,'query_id':q.query_id,'condition':groups[q.query_id],
             'machine_type':q.machine_type,'selected_error':errors[0],'baseline_error':errors[1],
             'error_difference':errors[0]-errors[1]}
        rows.append(row); episodes[(groups[q.query_id],q.episode_id)].append(errors)
        types[(q.machine_type,q.episode_id)].append(errors)
        for i,payout in enumerate((0,1,2,5,20)):
            payouts[payout].append(((selected[t.query_id][i]-t.probabilities[i])**2,
                                   (baseline[t.query_id][i]-t.probabilities[i])**2))
    clusters=defaultdict(list)
    episode_rows=[]
    for (condition,episode),values in episodes.items():
        averages=[mean(v[i] for v in values) for i in range(3)]
        clusters[condition].append(averages)
        episode_rows.append({'episode':episode,'condition':condition,'selected_error':averages[0],
                             'baseline_error':averages[1],'difference':averages[0]-averages[1]})
    def delta(samples):
        return mean((mean(v[0] for v in samples[c])-mean(v[1] for v in samples[c])) / max(mean(v[2] for v in samples[c]),1e-9) for c in CONDITIONS)
    rng=random.Random(20261107)
    boot=sorted(delta({c:rng.choices(clusters[c],k=len(clusters[c])) for c in CONDITIONS}) for _ in range(2000))
    report={'scope':'Tuning episodes only; post-selection diagnostics, not independent statistical confirmation',
        'episodes':len(episode_rows),'queries':len(rows),'normalized_loss_difference':delta(clusters),
        'paired_episode_bootstrap_95_percent_interval':[boot[49],boot[1949]],
        'bootstrap_seed':20261107,'bootstrap_repeats':2000,
        'episodes_improved':sum(r['difference']<0 for r in episode_rows),
        'by_condition':{c:{'selected_error':mean(v[0] for v in vs),'baseline_error':mean(v[1] for v in vs),
                           'error_difference':mean(v[0]-v[1] for v in vs)} for c,vs in clusters.items()},
        'by_payout':{str(k):{'selected_component_mse':mean(v[0] for v in vs),'baseline_component_mse':mean(v[1] for v in vs)} for k,vs in payouts.items()},
        'by_type':{kind:{'selected_error':mean(mean(v[0] for v in vs) for (k,e),vs in types.items() if k==kind),
                         'baseline_error':mean(mean(v[1] for v in vs) for (k,e),vs in types.items() if k==kind)} for kind in sorted({k for k,e in types})},
        'largest_regressions':sorted(episode_rows,key=lambda r:r['difference'],reverse=True)[:8]}
    write_json(a.out/'report.json',report)
    with (a.out/'query_errors.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # A train-only development protocol for the next increment. No validation targets enter these folds.
    train_q=read_queries(a.data/'train/queries.csv')
    train_groups={r['query_id']:r['condition'] for r in read_rows(a.data/'train/conditions.csv',('query_id','condition'))}
    by_condition=defaultdict(set)
    for q in train_q: by_condition[train_groups[q.query_id]].add(q.episode_id)
    dev=[[] for _ in range(3)]
    for condition in CONDITIONS:
        ordered=sorted(by_condition[condition],key=lambda e:hashlib.sha256(('house-edge-cv-v1:'+e).encode()).hexdigest())
        for i,e in enumerate(ordered): dev[i%3].append(e)
    if len(set().union(*map(set,dev))) != sum(map(len,dev)): raise ValueError('CV episode overlap')
    write_json(a.out/'train_cv_protocol.json',{'folds':[sorted(x) for x in dev],
        'selection_data':'Only organizer training episodes; score held-out terminal observed outcomes',
        'objective':'Equal-condition, equal-episode observed-outcome Brier loss; noisy proxy, not official true-probability score',
        'rules':['No historical samples from a held-out episode may enter fitting.',
                 'Keep all candidate sets fixed before cross-validation.',
                 'Do not tune against the already-inspected audit.',
                 'Validation after selection is descriptive and is not a new independent audit.']})
    print(json.dumps(report,indent=2))
if __name__ == '__main__': main()
