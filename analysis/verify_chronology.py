"""Verify the original transfer-study receipts and reconstruct eight prediction CSV hashes."""
from datetime import datetime
from pathlib import Path
import csv
import gzip
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "provenance"
TABLES = ROOT / "tables"
DATA = ROOT / "data"
SPLITS = ("calibration", "test")
VARIANTS = ("variant1", "variant2")
WINDOWS = ("events30k", "time20")


def load(path):
    return json.loads(path.read_bytes())


def require(condition, description):
    if not condition:
        raise ValueError(description)


def verify_digest(path, expected):
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    require(actual == expected, f"SHA-256 mismatch for {path.name}")


def prediction_digests(variant, window, runs):
    """Split the losslessly combined CSV at its split field, preserving original bytes."""
    with gzip.open(DATA / f"{variant}_{window}.csv.gz", "rb") as stream:
        header = next(stream)
        column = header.rstrip().split(b",").index(b"split")
        digests = {split: hashlib.sha256(header) for split in SPLITS}
        counts = dict.fromkeys(SPLITS, 0)
        for line in stream:
            split = line.split(b",", column + 1)[column].decode("ascii")
            digests[split].update(line)
            counts[split] += 1
    for split in SPLITS:
        run = runs[variant, split]
        expected = run["outputs_sha256"][f"{window}/predictions.csv"]
        require(digests[split].hexdigest() == expected and counts[split] == run["n"],
                f"Original prediction file mismatch for {variant}/{split}/{window}")


def main():
    protocol = load(RECORDS / "protocol_receipt.json")
    inputs = load(RECORDS / "input_checks.json")
    parity = load(RECORDS / "new_parity.json")
    forecast_file = TABLES / "trajectory_forecast.json"
    forecast = load(forecast_file)
    receipt_file = TABLES / "trajectory_forecast_receipt.json"
    receipt = load(receipt_file)
    release_file = RECORDS / "test_release.json"
    release = load(release_file)
    audit = load(RECORDS / "inference_verification.json")
    calibration_check = RECORDS / "sealed_forecast_verification.json"

    for sha in (protocol["sha256"]["revision-transfer/FROZEN_PROTOCOL.md"],
                forecast["input_sha256"]["revision-transfer/FROZEN_PROTOCOL.md"],
                receipt["frozen_protocol_sha256"]):
        verify_digest(RECORDS / "FROZEN_PROTOCOL.md", sha)
    verify_digest(RECORDS / "protocol_receipt.json", inputs["protocol_receipt_sha256"])
    verify_digest(RECORDS / "protocol_receipt.json",
                  forecast["input_sha256"]["revision-transfer/protocol_receipt.json"])
    verify_digest(RECORDS / "input_checks.json", parity["input_checks_sha256"])
    for sha in (release["receipt_sha256"], audit["forecast_receipt_sha256"]):
        verify_digest(receipt_file, sha)
    for sha in (receipt["forecast_sha256"], release["forecast_sha256"]):
        verify_digest(forecast_file, sha)
    verify_digest(TABLES / "trajectory_forecast_draws.npz", receipt["draws_sha256"])
    verify_digest(calibration_check, release["independent_calibration_check_sha256"])
    verification = load(calibration_check)
    require(verification["pass"] and verification["calibration_only"],
            "Original forecast verification did not pass on calibration alone")
    verify_digest(release_file, audit["test_release_sha256"])
    require(parity["archive_calibration_only"] and len(parity["cases"]) == 24
            and all(c["tensor_max_abs"] == c["logit_max_abs"] == 0 for c in parity["cases"]),
            "Calibration parity check failed")

    with (DATA / "input_releases.csv").open(newline="") as stream:
        inventory = {r["condition_or_model"]: r["sha256"] for r in csv.DictReader(stream)
                     if r["model_set"] == "full"}
    for model, checkpoint in inputs["checkpoints"].items():
        require(checkpoint["weight_sha256"] == inventory[model], f"{model} checkpoint hash")
    runs = {}
    for entry in audit["runs"]:
        variant, split = entry["variant"], entry["split"]
        path = RECORDS / f"{variant}_{split}.json"
        verify_digest(path, entry["run_receipt_sha256"])
        run = load(path)
        require((run["variant"], run["split"]) == (variant, split), "Run identity mismatch")
        require(run["archive_sha256"] == inventory[variant], "Archive hash mismatch")
        require(run["n"] == entry["source_ids_checked"] == inputs[f"{split}_sources"],
                "Split count mismatch")
        require(datetime.fromisoformat(run["started_utc"]) <= datetime.fromisoformat(run["finished_utc"]),
                "Run finish precedes start")
        refs = run["references"]
        verify_digest(RECORDS / "protocol_receipt.json", refs["protocol_receipt_sha256"])
        verify_digest(RECORDS / "new_parity.json", refs["sha256_new_parity"])
        if split == "test":
            verify_digest(receipt_file, refs["forecast"]["sha256"])
            verify_digest(release_file, refs["forecast"]["release_sha256"])
            require(refs["forecast"]["release_utc"] == release["release_utc"], "Release time mismatch")
        else:
            require(refs["forecast"] is None, "Calibration receipt specifies test authorization")
            key = f"revision-transfer/outputs/calibration/{variant}/events30k/predictions.csv"
            require(forecast["input_sha256"][key] == run["outputs_sha256"]["events30k/predictions.csv"],
                    "Forecast input differs from calibration run")
        runs[variant, split] = run

    require(set(runs) == {(v, s) for v in VARIANTS for s in SPLITS}, "Missing run")
    for variant in VARIANTS:
        for window in WINDOWS:
            prediction_digests(variant, window, runs)
    stages = [protocol["utc"], inputs["utc"],
              min(runs[v, "calibration"]["started_utc"] for v in VARIANTS),
              max(runs[v, "calibration"]["finished_utc"] for v in VARIANTS),
              forecast["created_utc"], receipt["created_utc"], release["release_utc"],
              min(runs[v, "test"]["started_utc"] for v in VARIANTS),
              max(runs[v, "test"]["finished_utc"] for v in VARIANTS), audit["verified_utc"]]
    times = [datetime.fromisoformat(t) for t in stages]
    require(all(a < b for a, b in zip(times, times[1:])), "Inconsistent recorded chronology")
    print("Verified the frozen protocol, sealed forecasts and authorization, four run receipts, "
          "and eight original prediction files. Recorded forecast seal and first test start (UTC):",
          receipt["created_utc"], stages[7])


if __name__ == "__main__":
    main()
