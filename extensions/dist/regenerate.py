"""Regenerate DiST logits and their row index from an official N-ImageNet archive."""
import argparse
import csv
import ctypes
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from torchvision.models import resnet34

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "regeneration"))
from infer import csv_rows, recording_key, sha256

N = 30000
NAMES = {"original": ("hash","first"), "vertical": ("hash",), "horizontal": ("hash",),
         "variant1": ("first","second"), "variant2": ("first",)}

def build_encoder(temp):
    library = temp / "encode.so"
    subprocess.run(["g++", "-std=c++17", "-O3", "-ffp-contract=off", "-fPIC", "-shared",
                    str(HERE/"encode.cpp"), "-o", str(library)], check=True)
    lib = ctypes.CDLL(str(library))
    arg = lambda t: np.ctypeslib.ndpointer(dtype=t, flags=("C_CONTIGUOUS",))
    lib.encode_dist.argtypes = [arg(np.int32), arg(np.float64), ctypes.c_int, arg(np.float32)]
    lib.encode_dist.restype = ctypes.c_int
    return lib

def model(checkpoint):
    m=resnet34(weights=None,num_classes=1000)
    m.conv1=torch.nn.Conv2d(2,64,14,stride=2,padding=3,bias=False)
    ck=torch.load(checkpoint,map_location="cpu",weights_only=True)
    state=ck["state_dict"]
    if all(k.startswith("module.") for k in state):
        state={k.removeprefix("module."):v for k,v in state.items()}
    m.load_state_dict(state,strict=True)
    return m.eval()

def bounds(n,key,name):
    if name=="hash":
        start=int.from_bytes(hashlib.sha256(("crop/"+key).encode()).digest()[:8],"big")*max(1,n-N+1)>>64
    elif name=="first":
        start=0
    else:
        start=min(n,N)
    return start,min(n,start+N)

def encode(blob,key,names,lib):
    with np.load(io.BytesIO(blob),allow_pickle=False) as ar:
        ev=ar["event_data"].reshape(-1)
    assert len(ev) and ev.dtype["p"]==np.bool_
    result,metadata=[],[len(ev)]
    for name in names:
        a,b=bounds(len(ev),key,name)
        block=ev[a:b]
        times=block["t"].astype(np.float64)/1e6
        x=block["x"]
        y=block["y"]
        assert np.all(x<640) and np.all(y<480)
        assert len(times)<2 or np.all(times[1:]>=times[:-1]), (key,name)
        idx=(x*(224/640)).astype(np.int32)+224*(y*(224/480)).astype(np.int32)+(~block["p"]).astype(np.int32)*(224*224)
        out=np.empty((2,224,224),np.float32)
        status=lib.encode_dist(idx,times,len(block),out)
        assert status==0, (key,name,a,b)
        metadata.extend((a,b,int(block["t"][0]) if len(block) else -1,int(block["t"][-1]) if len(block) else -1))
        result.append(out)
    return result,metadata

def check_file(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"SHA-256 mismatch for {path.name}")

def source_info():
    classes = {r["synset"]: int(r["label"]) for r in csv_rows(ROOT/"data/class_map.csv")}
    records = {}
    for r in csv_rows(ROOT/"data/source_index.csv"):
        key = r["source_id"]
        label = classes[key.split("/")[0]]
        if label != int(r["label"]) or key in records:
            raise ValueError(f"Class or source mismatch for {key}")
        records[key] = (label, r["split"] if r["cohort"] == "full" else "pilot_excluded")
    if len(records) != 50000:
        raise ValueError("Expected all 50,000 upstream validation sources")
    return records

@torch.inference_mode()
def run(args, temp, out):
    pins = json.loads((HERE/"provenance.json").read_text())
    archive_info = pins["archives"][args.condition]
    if args.archive.stat().st_size != archive_info["bytes"]:
        raise ValueError(f"Unexpected size for {args.archive.name}")
    for path, expected in ((args.archive,archive_info["sha256"]), (args.weights,pins["checkpoint_sha256"]),
                           (args.config,pins["configuration_sha256"]), (ROOT/"data/class_map.csv",pins["class_map_sha256"]),
                           (ROOT/"data/source_index.csv",pins["source_index_sha256"])):
        check_file(path,expected)
    records = source_info()
    net = model(args.weights).to(args.device)
    lib = build_encoder(temp)
    names = NAMES[args.condition]
    with ZipFile(args.archive) as z:
        entries = [(recording_key(e.filename), e) for e in z.infolist() if not e.is_dir()]
        if len(entries) != 50000 or set(k for k,_ in entries) != records.keys():
            raise ValueError("Archive source IDs differ from the published source mapping")
        entries = [(k,e) for k,e in entries if (args.condition == "original" or records[k][1] != "pilot_excluded")
                   and (args.source_id is None or k == args.source_id)]
        if not entries:
            raise ValueError("No matching source in this condition")
        fields = ["source_id","label","split","raw_events"]
        fields += [crop+"_"+field for crop in names for field in ("start","stop","first_us","last_us")]
        fields += [crop+"_"+field for crop in names for field in ("class","msp")]
        matrices = {crop: np.lib.format.open_memmap(out/(crop+".npy"),"w+",np.float32,
                    shape=(len(entries),1000)) for crop in names}
        correct_original = 0
        def encode_entry(pair):
            key,blob = pair
            return encode(blob,key,names,lib)
        with (out/"sources.csv").open("w",newline="") as f, ThreadPoolExecutor(args.workers) as pool:
            writer = csv.writer(f)
            writer.writerow(fields)
            for start in range(0,len(entries),args.batch):
                batch = entries[start:start+args.batch]
                encoded = list(pool.map(encode_entry, [(k,z.read(e)) for k,e in batch]))
                columns = []
                for j,crop in enumerate(names):
                    x = torch.from_numpy(np.stack([views[j] for views,meta in encoded])).to(args.device)
                    logits = net(x).cpu().numpy()
                    if logits.shape != (len(batch),1000) or not np.isfinite(logits).all():
                        raise ValueError(f"Invalid logits at row {start} for {crop}")
                    matrices[crop][start:start+len(batch)] = logits
                    msp = 1/np.exp(logits.astype(np.float64)-logits.max(1,keepdims=True)).sum(1)
                    columns += [logits.argmax(1),msp]
                for j,((key,_),(_,meta)) in enumerate(zip(batch,encoded)):
                    label,split = records[key]
                    writer.writerow([key,label,split,*meta,*[v[j] for v in columns]])
                    if args.condition == "original":
                        correct_original += int(columns[0][j] == label)
        for matrix in matrices.values():
            matrix.flush()
    settings = dict(condition=args.condition,n=len(entries),batch=args.batch,device=str(args.device),
                    torch=str(torch.__version__),archive_sha256=archive_info["sha256"])
    if args.condition == "original" and args.source_id is None:
        pct = 100*correct_original/len(entries)
        settings["gate"] = dict(correct=correct_original,measured_pct=pct,published_pct=48.43)
        if abs(pct-48.43)>1:
            raise ValueError(f"Original-condition accuracy {pct} failed the one-point verification gate.")
    (out/"settings.json").write_text(json.dumps(settings,indent=2)+"\n")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("condition",choices=NAMES)
    parser.add_argument("--archive",type=Path,required=True)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--weights",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--source-id")
    parser.add_argument("--batch",type=int,default=32)
    parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    args=parser.parse_args()
    if args.batch < 1 or args.workers < 1 or args.output.exists():
        parser.error("Specify a new output directory and positive batch/worker counts.")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as td:
        temp=Path(td)
        out=temp/"inference"
        out.mkdir()
        run(args,temp,out)
        out.rename(args.output)

if __name__=="__main__":
    main()
