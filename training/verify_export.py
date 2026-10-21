"""Verify fitted CUDA constants against checkpoint and actual supervised samples."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import torch
from model import MRPNN
from verify import read_trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("training/runs/physical-smoke"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = root/args.run
    report = {"atol": 1e-4, "rtol": 1e-4, "cases": []}
    fit = json.loads((run/"fit-report.json").read_text())
    assert fit["passed"]
    for name, digest in fit["artifacts_sha256"].items():
        assert hashlib.sha256((run/name).read_bytes()).hexdigest() == digest
    model = MRPNN()
    model.load_state_dict(torch.load(run/"weights.pt", weights_only=True))
    header_model = MRPNN(run/"NNWeight.cuh")
    assert all(torch.equal(v, header_model.state_dict()[k]) for k,v in model.state_dict().items())
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    env = os.environ.copy()
    env["PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;" + env["PATH"]
    replay = np.fromfile(run/"replay.bin", "<f4", offset=4).reshape(96, 20)
    cases = [("cloud_isotropic", "CLOUD03", 0., .5),
             ("cloud_forward", "CLOUD03", .857, 6.), ("bunny", "BUNNY", .5, 1.)]
    with tempfile.TemporaryDirectory(prefix="mrpnn-export-") as temp:
        for i, (name, scene, g, alpha) in enumerate(cases):
            inputs_path, output = Path(temp)/"input.bin", Path(temp)/"trace.bin"
            with open(inputs_path, "wb") as f:
                np.asarray([32], "<i4").tofile(f)
                replay[i*32:(i+1)*32].astype("<f4").tofile(f)
            process = subprocess.run([
                str(root/"build-5080/Release/FittedNetworkProbe.exe"),
                str(root/f"TestCase/{scene}"), str(output), "32", str(g),
                str(alpha), str(inputs_path)], cwd=root, env=env,
                capture_output=True, timeout=180)
            if process.returncode:
                raise RuntimeError(process.stdout.decode(errors="replace")+
                                   process.stderr.decode(errors="replace"))
            data = read_trace(output)
            source = np.load(root/f"training/datasets/smoke/{name}-4096.npz")
            features = {}
            for k in ("density", "transmittance", "phase", "g", "gamma", "albedo"):
                target = torch.from_numpy(source[k][:32].copy())
                features[k] = dict(max_abs=float((data[k]-target).abs().max()),
                                   passed=bool(torch.allclose(data[k], target, atol=1e-5, rtol=1e-5)))
            item = dict(name=name, supervised_features=features, devices={})
            for device in ("cpu", "cuda"):
                model.to(device)
                with torch.no_grad():
                    predicted = model(**{k:data[k].to(device) for k in features}, return_trace=True)
                stats = {}
                for k,v in predicted.items():
                    value = v.cpu()
                    stats[k] = dict(max_abs=float((value-data[k]).abs().max()),
                                    finite=bool(torch.isfinite(value).all()),
                                    passed=bool(torch.allclose(value,data[k],atol=1e-4,rtol=1e-4)))
                item["devices"][device] = stats
            report["cases"].append(item)
            print(name, json.dumps(item["devices"]["cuda"]["radiance"]), flush=True)
    report["header_matches_checkpoint"] = True
    report["passed"] = all(s["passed"] for c in report["cases"] for s in c["supervised_features"].values()) and all(
        s["passed"] and s["finite"] for c in report["cases"]
        for stages in c["devices"].values() for s in stages.values())
    (run/"export-verification.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("export passed =", report["passed"])
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
