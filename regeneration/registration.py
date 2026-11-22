"""Outcome-blind JPEG-to-sensor registration for the N-ImageNet orientation study."""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from itertools import product
from pathlib import Path
import argparse
import csv
import tempfile
import hashlib
import io
import json
import math
import zipfile

import numpy as np
from PIL import Image

import cv2


DATA = Path(__file__).resolve().parents[1] / "data"
cv2.setNumThreads(1)
RULES = ("fit", "fill", "stretch")
AXES = tuple(f"r{r}m{m}" for r in range(4) for m in (0,1))
BANDS = ("portrait","near_square","landscape")
DIRECTIONS = ("vertical","horizontal")
NATIVE = (640,480)
ANCHOR_WIDTH = 49
RADIUS = 32


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_table(path, rows):
    with path.open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def selection():
    with (DATA/"whole_source_features.csv").open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "calibration"]
    assert len(rows)==9000 and len({r["source_id"] for r in rows})==9000
    groups={b:[] for b in BANDS}
    for r in rows:
        assert r["split"]=="calibration"
        w,h=int(r["width_px"]),int(r["height_px"])
        band="portrait" if 10*w<9*h else "landscape" if 10*w>11*h else "near_square"
        key=r["source_id"]
        rank=hashlib.sha256(("geometry-20261121/"+key).encode()).hexdigest()
        groups[band].append((rank,key,w,h))
    chosen=[]
    for band, items in groups.items():
        assert len(items)>=40
        for i,(rank,key,w,h) in enumerate(sorted(items)[:40]):
            chosen.append(dict(source_id=key,band=band,aspect_rank=i+1,width=w,height=h,hash_order=rank,
                               fold="fit" if i<20 else "validation"))
    return chosen


def jpeg_gray(rgb):
    # Geometry uses log relative luminance. Directional gradients are subsequently
    # standardized locally and serve as geometric templates only.
    u=rgb.astype(np.float32)/255
    linear=np.where(u<=0.04045,u/12.92,((u+0.055)/1.055)**2.4)
    return np.log(np.maximum(linear@np.array([.2126,.7152,.0722],np.float32),1e-4))


def load_samples(chosen, rgb_directory, archive_paths):
    samples=[]
    archives=[zipfile.ZipFile(path) for path in archive_paths]
    try:
        maps=[{"/".join(Path(m.filename).parts[-2:]).removesuffix(".npz"):m for m in a.infolist()
               if m.filename.endswith(".npz")} for a in archives]
        for row in chosen:
            key=row["source_id"]
            with Image.open(rgb_directory/(key.split("/")[1]+".JPEG")) as im:
                assert im.size==(row["width"],row["height"])
                gray=jpeg_gray(np.array(im.convert("RGB")))
            activity=[]
            for a, table in zip(archives,maps):
                with np.load(io.BytesIO(a.read(table[key])),allow_pickle=False) as f:
                    ev=f["event_data"].reshape(-1)
                    x,y=ev["x"].astype(np.int32),ev["y"].astype(np.int32)
                    assert (x>=0).all() and (x<640).all() and (y>=0).all() and (y<480).all()
                    counts=np.bincount(y*640+x,minlength=640*480).reshape(480,640)
                activity.append(np.log1p(counts.astype(np.float32)))
            samples.append((row,gray,activity))
    finally:
        for a in archives:
            a.close()
    return samples


def orient(img, axis):
    out=np.rot90(img,int(axis[1]))
    return np.fliplr(out) if axis[3]=="1" else out


@lru_cache(maxsize=40)
def sensor_coords(width,roll,cx,cy):
    h=width*3//4
    yy,xx=np.indices((h,width),dtype=np.float32)
    xx-=(width-1)/2+cx*width/640
    yy-=(h-1)/2+cy*width/640
    c,s=np.cos(np.deg2rad(roll)),np.sin(np.deg2rad(roll))
    return c*xx+s*yy, -s*xx+c*yy


def render(image, width, model, offset, rgb=False, source_size=None):
    rule,axis,rw,rh,angle=model
    src=orient(image,axis)
    h,w=src.shape[:2]
    # Preserve exact JPEG aspect even if a decimated fitting pyramid rounded a side.
    ow,oh=source_size or (image.shape[1],image.shape[0])
    if int(axis[1])%2:
        ow,oh=oh,ow
    f=width/640
    rectw,recth=rw*640*f,rh*480*f
    if rule=="fit":
        mag=min(rectw/ow,recth/oh)
        sw,sh=ow*mag,oh*mag
    elif rule=="fill":
        mag=max(rectw/ow,recth/oh)
        sw,sh=ow*mag,oh*mag
    else:
        sw,sh=rectw,recth
    sx,sy=sw/w,sh/h
    c,s=np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))
    mat=np.array([[c*sx,-s*sy],[s*sx,c*sy]])
    center=np.array([(width-1)/2 + offset[0]*f,(width*3//4-1)/2 + offset[1]*f])
    shift=center-mat@np.array([(w-1)/2,(h-1)/2])
    warped=cv2.warpAffine(src,np.column_stack([mat,shift]),(width,width*3//4),
                          flags=cv2.INTER_LANCZOS4 if rgb else cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    gx,gy=sensor_coords(width,angle,*offset)
    valid=(abs(gx)<min(rectw,sw)/2) & (abs(gy)<min(recth,sh)/2)
    return warped, valid


def contrast(image,width):
    # Sparse events may blur the edge. The native low/high widths are scaled
    # with each pyramid level and bounded by a fraction of a pyramid pixel.
    sm=cv2.GaussianBlur(image,(0,0),max(.65,1.4*width/640))
    return sm-cv2.GaussianBlur(sm,(0,0),max(1.8,9*width/640))


def templates(image,source_size,model,width,offset):
    warped,valid=render(image,width,model,offset,source_size=source_size)
    valid=cv2.erode(valid.astype(np.uint8),np.ones((3,3),np.uint8),borderValue=0)
    warped=cv2.GaussianBlur(warped,(3,3),.65)
    out=[]
    for d in DIRECTIONS:
        dx=int(d=="horizontal")
        grad=np.abs(cv2.Sobel(warped,cv2.CV_32F,dx,1-dx,ksize=3))*valid
        out.append(contrast(grad,width)*valid)
    return out,valid.astype(bool)


def pyramid(samples,width):
    prepared=[]
    for row,gray,ev in samples:
        ow,oh=gray.shape[1],gray.shape[0]
        factor=min(1.0,2*width/max(ow,oh))
        src=cv2.resize(gray,(max(1,round(ow*factor)),max(1,round(oh*factor))),
                       interpolation=cv2.INTER_AREA) if factor<1 else gray
        targets=[contrast(cv2.resize(e,(width,width*3//4),interpolation=cv2.INTER_AREA),width) for e in ev]
        prepared.append((row,src,(ow,oh),targets))
    return prepared


def fit_one(args):
    model,dataset,width=args
    h=width*3//4
    px,py=width//8,h//8
    surfaces=np.zeros((2,2*py+1,2*px+1),np.float64)
    for _,src,ss,target in dataset:
        template,_=templates(src,ss,model,width,(0.,0.))
        for i,(t,e) in enumerate(zip(template,target)):
            crop=t[py:h-py,px:width-px]
            if crop.std()>1e-7:
                surfaces[i]+=cv2.matchTemplate(e,crop,cv2.TM_CCOEFF_NORMED)
    surfaces/=len(dataset)
    offsets={}
    vals=[]
    for d,sc in zip(DIRECTIONS,surfaces):
        by,bx=np.unravel_index(sc.argmax(),sc.shape)
        offsets[d]=[float((bx-px)*640/width),float((by-py)*640/width)]
        vals.append(float(sc[by,bx]))
    return dict(parameters=list(model),score=np.mean(vals).item(),directional_scores=dict(zip(DIRECTIONS,vals)),
                offsets_native_px=offsets)


def fitting(samples,workers,out):
    log=[]
    fit=[r for r in samples if r[0]["fold"]=="fit"]
    def stage(label,models,width,items,families):
        prepared=pyramid(items,width)
        args=[(m,prepared,width) for m in models]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            allr=list(pool.map(fit_one,args))
        ranked=sorted(allr,key=lambda r:-r["score"])
        kept=[]
        seen=set()
        for r in ranked:
            pair=tuple(r["parameters"][:2])
            if pair not in seen:
                kept.append(r)
                seen.add(pair)
            if len(kept)>=families:
                break
        log.append(dict(stage=label,width=width,n_sources=len(items),n_candidates=len(models),leading=kept))
        print(f"{label}: {len(models)} candidates, {len(items)} sources; best geometry score {kept[0]['score']:.4f}",flush=True)
        return kept
    scale=[.4,.6,.8,1.,1.25,1.5,2.]
    base=[(*x,0.) for x in product(RULES,AXES,scale,scale)]
    best=stage("coarse",base,96,[r for r in fit if r[0]["aspect_rank"]<=4],6)
    for label,width,factors,rolls,nkeepers in [
        ("dimensions",160,[.85,1,1.18],[0.],2),
        ("roll",224,[.94,1,1.06],[-3.,0.,3.],1),
        ("fine",320,[.975,1,1.025],[-1.5,0.,1.5],1)]:
        candidates=[]
        for top in best:
            rule,axis,ww,hh,roll=top["parameters"]
            candidates.extend((rule,axis,ww*fw,hh*fh,roll+dr) for fw,fh,dr in product(factors,factors,rolls))
        best=stage(label,candidates,width,fit,nkeepers)
    best=stage("native_translation",[tuple(best[0]["parameters"])],640,fit,1)
    save_json(out / "fit.json",dict(stages=log,selected=best[0]))
    return best[0]


def landmark_trials(row,source,size,event,which,model):
    direction=DIRECTIONS[which]
    ts,valid=templates(source,size,model["parameters"],640,tuple(model["offsets_native_px"][direction]))
    template=ts[which]
    half=ANCHOR_WIDTH//2
    margin=half+RADIUS
    good=cv2.erode(valid.astype(np.uint8),np.ones((2*half+1,2*half+1),np.uint8),borderValue=0).astype(bool)
    good[:margin]=False
    good[-margin:]=False
    good[:,:margin]=False
    good[:,-margin:]=False
    corner=cv2.cornerMinEigenVal(template,blockSize=7,ksize=3)
    yy,xx=np.where(good)
    if len(xx):
        xlo,xhi,ylo,yhi=xx.min(),xx.max()+1,yy.min(),yy.max()+1
    else:
        xlo=xhi=ylo=yhi=0
    items=[]
    for i in range(4):
        half_x=i%2
        half_y=i//2
        xa=int(xlo+(xhi-xlo)*half_x/2)
        xb=int(xlo+(xhi-xlo)*(half_x+1)/2)
        ya=int(ylo+(yhi-ylo)*half_y/2)
        yb=int(ylo+(yhi-ylo)*(half_y+1)/2)
        rec=dict(source_id=row["source_id"],band=row["band"],direction=direction,quadrant=i,
                 source_x="",source_y="",match_x="",match_y="",ncc="",second_peak="",error_px="",
                 reason="unavailable")
        items.append(rec)
        if xa>=xb or ya>=yb or not good[ya:yb,xa:xb].any():
            continue
        scores=np.where(good[ya:yb,xa:xb],corner[ya:yb,xa:xb],-np.inf)
        iy,ix=np.unravel_index(scores.argmax(),scores.shape)
        x,y=int(ix+xa),int(iy+ya)
        rec.update(source_x=x,source_y=y)
        patch=template[y-half:y+half+1,x-half:x+half+1]
        if scores[iy,ix]<=1e-8 or patch.std()<=1e-6:
            rec["reason"]="untextured"
            continue
        area=event[y-margin:y+margin+1,x-margin:x+margin+1]
        corr=cv2.matchTemplate(area,patch,cv2.TM_CCOEFF_NORMED)
        py,px=np.unravel_index(corr.argmax(),corr.shape)
        dx,dy=int(px-RADIUS),int(py-RADIUS)
        ry,rx=np.indices(corr.shape)
        peak=float(corr[py,px])
        second=float(corr[(ry-py)**2+(rx-px)**2>25].max())
        rec.update(match_x=x+dx,match_y=y+dy,ncc=peak,second_peak=second)
        if abs(dx)==RADIUS or abs(dy)==RADIUS:
            rec["reason"]="search_boundary"
        elif peak<.25:
            rec["reason"]="weak"
        elif peak-second<.03:
            rec["reason"]="ambiguous"
        else:
            rec.update(error_px=math.hypot(dx,dy),reason="matched")
    return items


def aggregate(rows):
    errors=sorted(float(r["error_px"]) if r["reason"]=="matched" else math.inf for r in rows)
    finite=[x for x in errors if math.isfinite(x)]
    p=errors[math.ceil(.9*len(rows))-1]
    reasons={s:sum(r["reason"]==s for r in rows) for s in sorted({r["reason"] for r in rows})}
    return dict(scheduled=len(rows),matched=len(finite),within_5px=sum(e<=5 for e in finite),by_status=reasons,
                p90_including_failures_px=p if math.isfinite(p) else None,
                p90_unbounded=not math.isfinite(p),finite_median_p90_px=np.percentile(finite,[50,90]).tolist() if finite else None,
                meets_gate=bool(p<=5))


def validate(samples,model,out):
    trials=[]
    for row,source,size,event in pyramid([s for s in samples if s[0]["fold"]=="validation"],640):
        for i,d in enumerate(DIRECTIONS):
            trials.extend(landmark_trials(row,source,size,event[i],i,model))
    write_table(out/"validation_landmarks.csv",trials)
    summary={d:aggregate([r for r in trials if r["direction"]==d]) for d in DIRECTIONS}
    bands={d:{b:aggregate([r for r in trials if r["direction"]==d and r["band"]==b]) for b in BANDS} for d in DIRECTIONS}
    assert len(trials)==480 and all(s["scheduled"]==240 for s in summary.values())
    return summary,bands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Directory containing the original validation JPEGs")
    parser.add_argument("--vertical", type=Path, required=True, help="Official full vertical event ZIP")
    parser.add_argument("--horizontal", type=Path, required=True, help="Official full horizontal event ZIP")
    parser.add_argument("--output", type=Path, required=True, help="New directory for results")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1 or args.output.exists():
        parser.error("Specify a positive worker count and a new output directory")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as scratch:
        stage = Path(scratch)/"geometry"
        stage.mkdir()
        selected = selection()
        write_table(stage/"selection.csv", selected)
        samples = load_samples(selected, args.images, (args.vertical, args.horizontal))
        model = fitting(samples, args.workers, stage)
        summary, bands = validate(samples, model, stage)
        passed = all(entry["meets_gate"] for entry in summary.values())
        result = {"status": "passed" if passed else "failed", "model": model,
                  "verification": summary, "verification_by_aspect": bands,
                  "parameter_order": ["display_rule", "axis", "width_fraction_of_640", "height_fraction_of_480", "roll_deg"],
                  "gate": "Nearest-rank 90th percentile <= 5 pixels for both directions, counting unresolved landmarks as infinite",
                  "fit_examples": 60, "validation_examples": 60}
        save_json(stage/"registration.json", result)
        stage.rename(args.output)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
