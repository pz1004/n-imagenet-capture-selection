"""Recompute the prospective forecasts and fixed-threshold transfer from paired scores."""
import argparse
import json
from pathlib import Path

import numpy as np

from capture_policies import choose_vertical, fit_discordant, read
from trajectory_data import B, DATA, MODELS, TABLES, ids_for, pair_data, samples, scan, threshold


def old_discordant_accuracy(old, draw):
    x,v,h = old
    discordant = draw[v[draw] != h[draw]]
    folds = discordant % 5
    total = 0
    for fold in range(5):
        train = discordant[folds != fold]
        valid = discordant[folds == fold]
        params = fit_discordant(x[train],v[train],h[train])
        correct = np.where(choose_vertical(x[valid],params),v[valid],h[valid])
        total += int(correct.sum())
    return total / len(discordant)


def locked_target(target, cut):
    x,v,h = target
    return np.where(x >= -cut, v, h)


def forecast(old, new, cut, draw):
    a = old_discordant_accuracy(old,draw)
    _, v, h = new
    discord = (v[draw] != h[draw]).mean()
    first_only = (v[draw] & ~h[draw]).mean()
    correct = locked_target(new,cut)
    transported = 100*(discord*a-first_only)
    direct = 100*(correct[draw].mean()-v[draw].mean())
    return np.array([transported,direct])


def ci(draw):
    return np.percentile(draw,[2.5,97.5],axis=0).T


def pair_gain(eval_data, cut, draw):
    x,v,h=eval_data
    select_h = x[draw] < -cut
    rescues = np.count_nonzero(select_h & ~v[draw] & h[draw])
    losses = np.count_nonzero(select_h & v[draw] & ~h[draw])
    return 100*(rescues-losses)/len(draw)


def evaluate(model, cal_ids, test_ids, full_draws):
    old = pair_data("old","calibration",model,cal_ids)
    new = pair_data("new","calibration",model,cal_ids)
    held = pair_data("new","test",model,test_ids)
    cut = threshold(model)
    cal_ix = np.arange(len(cal_ids))
    test_ix = np.arange(len(test_ids))
    predicted = forecast(old,new,cut,cal_ix)
    observed = pair_gain(held,cut,test_ix)
    published = json.loads((TABLES/"trajectory_forecast.json").read_text())["models"][model]
    np.testing.assert_allclose(predicted,[published["transported_pp"],published["direct_pp"]],rtol=0,atol=1e-10)
    saved = json.loads((TABLES/"trajectory_gains.json").read_text())["model_windows"]["events30k_"+model]
    expected_gain=saved["contrasts"]["frozen_vs_first"]
    np.testing.assert_allclose(observed,expected_gain["pp"],rtol=0,atol=1e-10)
    report = {"threshold":cut,"forecast_methods":["transported","direct"],
              "predicted_pp":predicted.tolist(),"observed_pp":observed,
              "forecast_minus_observed_pp":(predicted-observed).tolist()}
    if full_draws:
        for design in ("within","class"):
            forecasts=np.empty((B,2))
            gains=np.empty(B)
            for i,(cal,test) in enumerate(zip(samples(design,True),samples(design,False))):
                forecasts[i]=forecast(old,new,cut,cal)
                gains[i]=pair_gain(held,cut,test)
            expected=np.array([published[f"{m}_{design}_95"] for m in ("transported","direct")])
            np.testing.assert_allclose(ci(forecasts),expected,rtol=0,atol=1e-10)
            np.testing.assert_allclose(ci(gains),expected_gain[design+"_95_pp"],rtol=0,atol=1e-10)
            for j,method in enumerate(("transported","direct")):
                expected_error=saved["forecast"][method][design+"_error_95_pp"]
                np.testing.assert_allclose(ci(forecasts[:,j]-gains),expected_error,rtol=0,atol=1e-10)
            report[design]={"forecast_ci95_pp":ci(forecasts).tolist(),"gain_ci95_pp":ci(gains).tolist(),
                            "forecast_minus_observed_ci95_pp":ci(forecasts-gains[:,None]).tolist()}
    return report


def policy_table(model, window, cal_ids, test_ids, summary, full_draws):
    xc,vc,hc=pair_data("new","calibration",model,cal_ids,window)
    x,v,h=pair_data("new","test",model,test_ids,window)
    local=choose_vertical(x,fit_discordant(xc,vc,hc))
    outcomes={"first":v, "second":h, "best_cal_single":v if vc.sum()>=hc.sum() else h,
              "raw":np.where(x>=0,v,h), "frozen":np.where(x>=-threshold(model),v,h),
              "local":np.where(local,v,h), "oracle":v|h}
    if window=="events30k":
        sq=read(DATA/"original_sanity_predictions.csv","model_set","full")
        outcomes["square"]=np.array([sq[k][model+"_pred"]==sq[k]["label"] for k in test_ids])
    result={}
    for name,correct in outcomes.items():
        matches=int(correct.sum())
        assert matches==summary["accuracy"][name]["n_correct"]
        result[name]=matches
    for key,row in summary["contrasts"].items():
        selected,reference=key.split("_vs_")
        a,b=outcomes[selected],outcomes[reference]
        assert int((a&~b).sum())==row["rescues"] and int((b&~a).sum())==row["losses"]
    if full_draws:
        for design in ("within","class"):
            draws={name:np.empty(B) for name in outcomes}
            rank=np.empty((B,3))
            for i,(cal,test) in enumerate(zip(samples(design,True),samples(design,False))):
                best=v if vc[cal].sum()>=hc[cal].sum() else h
                fitted=fit_discordant(xc[cal],vc[cal],hc[cal])
                keep=choose_vertical(x[test],fitted)
                for name,correct in outcomes.items():
                    if name=="local":
                        score=np.where(keep,v[test],h[test])
                    elif name=="best_cal_single":
                        score=best[test]
                    else:
                        score=correct[test]
                    draws[name][i]=100*score.mean()
                vt,ht,xt=v[test],h[test],x[test]
                vx=xt[vt&~ht]; hx=xt[ht&~vt]
                rv=((vx>0).sum()+.5*(vx==0).sum())/len(vx)
                rh=((hx<0).sum()+.5*(hx==0).sum())/len(hx)
                rank[i]=100*np.array((rv,rh,(rv+rh)/2))
            np.testing.assert_allclose(ci(rank),summary["raw_ranking"][design+"_95_pct"],rtol=0,atol=1e-10)
            for name,values in draws.items():
                np.testing.assert_allclose(ci(values),summary["accuracy"][name][design+"_95_pct"],rtol=0,atol=1e-10)
            for key,contrast in summary["contrasts"].items():
                a,b=key.split("_vs_")
                np.testing.assert_allclose(ci(draws[a]-draws[b]),contrast[design+"_95_pp"],rtol=0,atol=1e-10)
    return result


def effect_change(model, window, cal_ids, test_ids, reference, full_draws):
    old_conf=scan("old","calibration","vertical",window,model,cal_ids)[0]
    gate=np.quantile(old_conf,.30,method="higher")
    arms=[]
    for pair in ("old","new"):
        score=scan(pair,"test","vertical",window,model,test_ids)[0]
        x,v,h=pair_data(pair,"test",model,test_ids,window)
        delta=(np.where(x>=-threshold(model),v,h).astype(int)-v.astype(int))
        arms.append((delta,score<gate))
    def effect(data,i):
        d,requests=data
        d=d[i]; g=requests[i]
        return np.array([d.mean(),(d*g).mean()-g.mean()*d.mean()])*100
    def paired(i):
        return effect(arms[1],i)-effect(arms[0],i)
    observed=paired(np.arange(len(test_ids)))
    np.testing.assert_allclose(observed,reference["transfer_minus_reference_pp"],rtol=0,atol=1e-10)
    if full_draws:
        for design in ("within","class"):
            limits=ci(np.asarray([paired(i) for i in samples(design,False)]))
            np.testing.assert_allclose(limits,reference[design+"_95_pp"],rtol=0,atol=1e-10)
    return observed.tolist()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap",action="store_true",help="Recalculate the 2,000 draws per design")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    cal_ids,test_ids=(ids_for(split) for split in ("calibration","test"))
    gains=json.loads((TABLES/"trajectory_gains.json").read_text())["model_windows"]
    changes=json.loads((TABLES/"trajectory_effect_changes.json").read_text())["results"]
    report={"forecast":{}, "correct_counts":{}, "transfer_minus_reference_effects_pp":{}}
    for model in MODELS:
        report["forecast"][model]=evaluate(model,cal_ids,test_ids,args.bootstrap)
        for window in ("events30k","time20"):
            key=window+"_"+model
            report["transfer_minus_reference_effects_pp"][key]=effect_change(model,window,cal_ids,test_ids,changes[key],args.bootstrap)
            report["correct_counts"][key]=policy_table(model,window,cal_ids,test_ids,gains[key],args.bootstrap)
    output=json.dumps(report,indent=2,allow_nan=False)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(output)
    else:
        print(output,end="")


if __name__=="__main__":
    main()
