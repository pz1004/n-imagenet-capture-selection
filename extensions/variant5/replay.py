"""Freeze calibration fits, then compare the held-out motion paths at equal request budgets."""
import argparse
import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from acquisition_inputs import source_groups, write_json
from capture_policies import choose_vertical, fit_discordant

MODELS = ('histogram','timestamp')
PATHS = ('horizontal','variant5','original')
GATES = ('quota','cutoff')
RULES = ('accept','local')
CONTRASTS = (('variant5','horizontal'),('original','variant5'),('original','horizontal'))
B = 2000




def mask_receipt(ids, mask):
    ordered=''.join(k+'\n' for k,take in zip(ids,mask) if take)
    return {'n':int(mask.sum()),'source_ids_sha256':hashlib.sha256(ordered.encode()).hexdigest()}


def cohort(table, split, groups):
    ids, labels = groups[split]
    data = {name: {m: (table[f"{split}_{name}_{m}_class"], table[f"{split}_{name}_{m}_msp"])
                   for m in MODELS} for name in ("vertical", *PATHS)}
    return ids, labels, data


def calibration_rules(model, calibration, test):
    _, y, captures = calibration
    ids, _, heldout = test
    first, msp = captures["vertical"][model]
    fits = {}
    for path in PATHS:
        second, second_msp = captures[path][model]
        fits[path] = list(fit_discordant(msp - second_msp, first == y, second == y))
    cutoff = float(np.quantile(msp, .30, method="higher"))
    first_test_msp = heldout["vertical"][model][1]
    return {"cutoff": cutoff, "fits": fits, "test_masks": {
        "quota": mask_receipt(ids, quota(first_test_msp, np.arange(len(ids)))),
        "cutoff": mask_receipt(ids, first_test_msp < cutoff)}}

def quota(scores, ids):
    k=round(.3*len(scores))
    boundary=np.partition(scores,k-1)[k-1]
    chosen=scores<boundary
    equal=np.flatnonzero(scores==boundary)
    take=k-chosen.sum()
    if take:
        chosen[equal[np.argsort(ids[equal],kind='stable')[:take]]]=True
    assert int(chosen.sum())==k
    return chosen


def arrays(groups,model,y):
    pred,ms=groups['vertical'][model]
    return (ms, pred==y, {path:(ms-groups[path][model][1], groups[path][model][0]==y) for path in PATHS})


def metrics(v, data, gates, fit):
    values={}
    counts={}
    correctness={}
    for path in PATHS:
        difference,second=data[path]
        values['alone/'+path]=100*second.mean()
        for rule in RULES:
            final=second if rule=='accept' else np.where(choose_vertical(difference,fit[path]),v,second)
            for gn,g in gates.items():
                key=f'{gn}/{rule}/{path}'
                take=np.where(g,final,v)
                corr=int(np.count_nonzero(take))
                gains=int(np.count_nonzero(take & ~v))
                losses=int(np.count_nonzero(~take & v))
                assert corr-v.sum()==gains-losses
                counts[key]={'correct':corr,'rescues':gains,'losses':losses,'requests':int(g.sum()),'n':len(v)}
                values[key+'/accuracy_pct']=100*corr/len(v)
                values[key+'/gain_pp']=100*(gains-losses)/len(v)
                values[key+'/yield_per100']=100*(gains-losses)/g.sum()
                random_accuracy=100*(v.mean()+g.mean()*(final.mean()-v.mean()))
                values[key+'/matched_random_accuracy_pct']=random_accuracy
                values[key+'/directed_minus_matched_random_pp']=values[key+'/accuracy_pct']-random_accuracy
                correctness[key]=take
    values['alone/vertical']=100*v.mean()
    for gn in GATES:
        for rule in RULES:
            for a,b in CONTRASTS:
                ca=correctness[f'{gn}/{rule}/{a}']
                cb=correctness[f'{gn}/{rule}/{b}']
                key=f'{gn}/{rule}/{a}_minus_{b}'
                counts[key]={'a_only':int(np.count_nonzero(ca&~cb)),'b_only':int(np.count_nonzero(cb&~ca))}
                values[key+'_pp']=100*(int(ca.sum())-int(cb.sum()))/len(v)
                random_difference=(values[f'{gn}/{rule}/{a}/matched_random_accuracy_pct']
                                   -values[f'{gn}/{rule}/{b}/matched_random_accuracy_pct'])
                values[key+'_beyond_random_pp']=values[key+'_pp']-random_difference
    return values,counts


def analysis(model,cal,test,rules,out):
    ids,y,tgr=test
    cy,cgr=cal[1],cal[2]
    cm,cv,cd=arrays(cgr,model,cy)
    tm,tv,td=arrays(tgr,model,y)
    gt={'quota':quota(tm,np.arange(len(y))), 'cutoff':tm<rules['cutoff']}
    assert {k:mask_receipt(ids,v) for k,v in gt.items()} == rules['test_masks']
    point,counts=metrics(tv,td,gt,rules['fits'])
    stats={key:{'estimate':float(val)} for key,val in point.items()}
    with gzip.open(out/(model+'_decisions.csv.gz'),'wt',newline='') as f:
        cols={'source_id':ids,'label':y,'quota_req':gt['quota'].astype(int),'cutoff_req':gt['cutoff'].astype(int)}
        v=tgr['vertical'][model][0]
        cols['vertical']=v
        for name in PATHS:
            p,s=tgr[name][model]
            use_v=choose_vertical(tm-s,rules['fits'][name])
            cols[name]=p
            cols[name+'_local_if_requested']=np.where(use_v,v,p)
        w=csv.writer(f)
        w.writerow(cols)
        w.writerows(zip(*cols.values()))
    for design,seed in (('within',2026112605),('class',2026112606)):
        rng=np.random.default_rng(seed)
        draws={k:np.empty(B) for k in point}
        offc=np.arange(900)[:,None]*10
        offt=np.arange(900)[:,None]*40
        for draw in range(B):
            if design=='within':
                cix=(offc+rng.integers(10,size=(900,10))).ravel()
                tix=(offt+rng.integers(40,size=(900,40))).ravel()
            else:
                classes=rng.integers(900,size=(900,1))
                cix=(classes*10+np.arange(10)).ravel()
                tix=(classes*40+np.arange(40)).ravel()
            m=tm[tix]
            cutoff=float(np.quantile(cm[cix],.30,method='higher'))
            gates={'quota':quota(m,tix),'cutoff':m<cutoff}
            fits={path:fit_discordant(diff[cix],cv[cix],second[cix]) for path,(diff,second) in cd.items()}
            valid={path:(diff[tix],second[tix]) for path,(diff,second) in td.items()}
            values,_=metrics(tv[tix],valid,gates,fits)
            for key,v in values.items():
                draws[key][draw]=v
        for key,a in draws.items():
            stats[key][design+'_ci95']=np.percentile(a,[2.5,97.5]).tolist()
        print(model,design,'complete',flush=True)
    write_json(out/(model+'.json'),{'n':len(y),'classes':900,'replicates':B,'fit_sources':len(cy),'counts':counts,'stats':stats})
    print(model,stats['quota/accept/variant5_minus_horizontal_pp'],stats['quota/accept/original_minus_variant5_pp'],flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=HERE / "inputs/scores.npz")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.inputs, allow_pickle=False) as table:
        groups = source_groups(table)
        cal, test = (cohort(table, split, groups) for split in ("calibration", "test"))
    archived = json.loads((HERE / "calibration_rules.json").read_text())["rules"]
    reference_input = args.inputs.resolve() == (HERE / "inputs/scores.npz").resolve()
    for model in MODELS:
        rules = calibration_rules(model, cal, test)
        # Refit from calibration labels and verify the originally frozen mask.
        if reference_input:
            assert rules["test_masks"] == archived[model]["test_masks"]
            np.testing.assert_allclose(rules["cutoff"], archived[model]["cutoff"], rtol=0, atol=1e-14)
            for path in PATHS:
                np.testing.assert_allclose(rules["fits"][path], archived[model]["fits"][path], rtol=0, atol=1e-11)
        analysis(model, cal, test, rules, args.output)


if __name__ == "__main__":
    main()
