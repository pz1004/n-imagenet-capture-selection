"""Check the calibration objective against the pinned upstream trainer on the same logits."""

import argparse
import ast
from pathlib import Path
from types import SimpleNamespace

import torch
from regenerate import ev_loss, loss_components, sha256

TRAINER_SHA256 = "d57a23c56077d76a924dec972d83974d2c5465ea5bd456a3281e229801426174"


def check(upstream):
    entropy, kl = loss_components(upstream)
    path = upstream / "real_cnn_model/train/trainer.py"
    if sha256(path) != TRAINER_SHA256:
        raise ValueError("The upstream trainer does not match the pinned revision")
    # Run the original train method on a logit parameter without importing the training application.
    tree = ast.parse(path.read_text())
    trainer = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "EvTTACNNTrainer"
    )
    train = next(
        node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == "train"
    )
    namespace = {"torch": torch, "accuracy": lambda *args, **kwargs: (0.0, 0.0)}
    exec(compile(ast.Module(body=[train], type_ignores=[]), str(path), "exec"), namespace)

    batch = 16
    logits = torch.randn(
        4 * batch, 7, dtype=torch.float64, generator=torch.Generator().manual_seed(3)
    )
    logits[batch : batch + 3] = logits[0]
    upstream_logits = torch.nn.Parameter(logits.clone())
    local_logits = torch.nn.Parameter(logits.clone())
    trainer = SimpleNamespace(
        model_container=SimpleNamespace(
            set_train=lambda keys: None, infer=lambda *args: upstream_logits
        ),
        use_cuda=False,
        cfg=SimpleNamespace(num_sentry_augment=3, sentry_inconsistent_fill=0.0),
        optimizer=torch.optim.SGD([upstream_logits], lr=0.0),
        loss_func=entropy,
        kl_loss_func=kl,
    )
    inputs = dict(
        input_data=torch.zeros(batch, 1),
        augment_input_data=torch.zeros(3 * batch, 1),
        label=torch.zeros(batch, dtype=torch.long),
    )
    upstream_value, *_ = namespace["train"](trainer, inputs)
    local_value, n_agree = ev_loss(local_logits, batch, (entropy, kl))
    local_value.backward()
    value_error = abs(upstream_value - local_value.item())
    gradient_error = (upstream_logits.grad - local_logits.grad).abs().max().item()
    assert 0 < n_agree < batch and value_error < 1e-12 and gradient_error < 1e-12
    print(
        f"{n_agree}/{batch} consistent sources; absolute loss error {value_error:g}; maximum gradient error {gradient_error:g}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ev-tta", required=True, type=Path)
    check(parser.parse_args().ev_tta)
