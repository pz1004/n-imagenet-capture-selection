"""Compare the public DiST encoder and classifier with the pinned upstream implementation."""
import argparse
import ast
import csv
import json
import subprocess
import sys
import tempfile
from configparser import ConfigParser
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

import torch

from regenerate import HERE, ROOT, bounds, build_encoder, check_file, encode, model, recording_key


def config(args, listing):
    parsed = ConfigParser()
    parsed.read(args.config)
    def read_value(text):
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return text
    cfg = SimpleNamespace(**{key:read_value(value) for key,value in parsed["Default"].items()})
    cfg.label_map = str(args.mapping.resolve())
    cfg.train_file = cfg.val_file = str(listing)
    cfg.mode = "test"
    cfg.load_model = str(args.weights.resolve())
    return cfg


@torch.inference_mode()
def run(args, work):
    pins = json.loads((HERE/"provenance.json").read_text())
    check_file(args.config, pins["configuration_sha256"])
    check_file(args.weights, pins["checkpoint_sha256"])
    check_file(args.mapping, "a1f4ff394cbbb3c04bc656b3a8209059bdb81f9076c94f27e89008de79d731bd")
    code = args.upstream.resolve()
    git = ["git","-C",str(code)]
    if subprocess.check_output(git+["rev-parse","HEAD"],text=True).strip() != pins["upstream_code_revision"]:
        raise ValueError("Upstream revision differs from provenance.json")
    subprocess.run(git+["diff","--quiet","HEAD","--","base","real_cnn_model"],check=True)
    sys.path[:0] = [str(code),str(code/"real_cnn_model")]
    from data.imagenet import ImageNetDataset
    from models.model_container import CNNContainer
    expected = pins["calibration_parity"][args.condition]
    keys = list(dict.fromkeys(r["source_id"] for r in expected))
    with ZipFile(args.archive) as z:
        files = {recording_key(e.filename):e for e in z.infolist() if not e.is_dir()}
        for key in keys:
            local = work/(key+".npz")
            local.parent.mkdir(parents=True,exist_ok=True)
            local.write_bytes(z.read(files[key]))
    listing = work/"samples.txt"
    listing.write_text("".join(str(work/(k+".npz"))+"\n" for k in keys))
    cfg = config(args, listing)
    data = ImageNetDataset(cfg,mode="test")
    with (ROOT/"data/class_map.csv").open() as f:
        assert data.labels == [r["synset"] for r in csv.DictReader(f)]
    own = model(args.weights)
    reference = CNNContainer(cfg)
    header = torch.load(cfg.load_model,weights_only=True,map_location="meta")["state_dict"]
    prefixed = all(k.startswith("module.") for k in header)
    if prefixed:
        reference.models["model"] = torch.nn.DataParallel(reference.models["model"])
    with patch.object(torch,"load",partial(torch.load,map_location="cpu",weights_only=True)):
        reference.load_saved()
    if prefixed:
        reference.models["model"] = reference.models["model"].module
    reference.models["model"].cpu().eval()
    lib = build_encoder(work)
    measured = []
    for i,key in enumerate(keys):
        cases = [r for r in expected if r["source_id"] == key]
        names = [r["crop"] for r in cases]
        views,metadata = encode((work/(key+".npz")).read_bytes(),key,names,lib)
        for case,name,view in zip(cases,names,views):
            cfg.slice_method = "idx"
            cfg.slice_start,cfg.slice_end = bounds(metadata[0],key,name)
            upstream,label = data[i]
            assert label == case["label"]
            x = torch.from_numpy(view)[None].contiguous()
            rx = upstream[None].contiguous()
            torch.testing.assert_close(x,rx,atol=1e-7,rtol=0)
            y = own(x)
            ry = reference.infer("model",rx,True)
            torch.testing.assert_close(y,ry,atol=1e-4,rtol=0)
            assert y.argmax()==ry.argmax()
            measured.append(dict(source_id=key,crop=name,input_max_abs=(x-rx).abs().max().item(),
                                 cpu_logit_max_abs=(y-ry).abs().max().item()))
    print(json.dumps(measured,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("condition",choices=("original","vertical","horizontal","variant1","variant2"))
    for argument in ["upstream","archive","config","weights","mapping"]:
        parser.add_argument("--"+argument,type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as folder:
        run(args,Path(folder))
