"""Cache deterministic history features with episode provenance for grouped CV."""
from collections import Counter
import numpy as np
from .features import HistoryIndex

def build_examples(queries, history, labels, augment):
    index=HistoryIndex(history)
    examples=[]; outcomes=[]; weights=[]; episodes=[]; terminal=[]; ids=[]; cutoffs=[]
    for q in sorted(queries,key=lambda q:q.query_id):
        candidates=[(q.cutoff,labels[q.query_id],True)]
        eligible=[(t,y) for t,y in index.before(q.episode_id,q.machine_id,q.cutoff) if t>=64]
        if augment and eligible:
            positions=np.linspace(0,len(eligible)-1,min(augment,len(eligible)),dtype=int)
            candidates += [(eligible[i][0],eligible[i][1],False) for i in positions]
        for t,y,is_terminal in candidates:
            row=index.features(q,t); columns=sorted(row)
            examples.append([row[k] for k in columns]); outcomes.append(y)
            weights.append(1/len(candidates)); episodes.append(q.episode_id)
            terminal.append(is_terminal); ids.append(q.query_id); cutoffs.append(t)
    return {'X':np.asarray(examples),'y':np.asarray(outcomes),'weight':np.asarray(weights),
            'episode':np.asarray(episodes),'terminal':np.asarray(terminal),'query_id':np.asarray(ids),
            'cutoff':np.asarray(cutoffs),'columns':np.asarray(columns)}

def validate_folds(folds, episodes):
    flat=[e for fold in folds for e in fold]
    if len(flat)!=len(set(flat)) or set(flat)!=set(episodes) or any(not f for f in folds):
        raise ValueError('Folds must partition all training episodes exactly once')

def fold_masks(data, held_out):
    held=np.isin(data['episode'],list(held_out))
    fit=~held; evaluate=held & data['terminal']
    if not fit.any() or not evaluate.any(): raise ValueError('Empty fit or evaluation fold')
    assert not set(data['episode'][fit]) & set(data['episode'][evaluate])
    return fit,evaluate
