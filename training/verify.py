"""Run CUDA real-volume probes and compare a trainable PyTorch transcription."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import torch
from model import MRPNN

FIELDS = [
    ("g", 1), ("gamma", 1), ("albedo", 3), ("density", 192),
    ("transmittance", 192), ("phase", 192), ("combined", 120),
    ("pooled", 75), ("gates", 18), ("reduced", 48),
    ("residuals", 288), ("logits", 3), ("radiance", 3),
]


def read_trace(path):
    with open(path, "rb") as f:
        count = np.fromfile(f, dtype="<i4", count=1)[0]
        data = np.fromfile(f, dtype="<f4")
    stride = sum(n for _, n in FIELDS)
    data = data.reshape(count, stride)
    result, offset = {}, 0
    for name, length in FIELDS:
        result[name] = torch.from_numpy(data[:, offset:offset+length].copy())
        offset += length
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="training/verification.json")
    parser.add_argument("--samples", type=int, default=256)
    args = parser.parse_args()
    root = Path(__file__).parents[1]
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = MRPNN()
    report = {"torch": torch.__version__, "parameters": sum(p.numel() for p in model.parameters()),
              "samples_per_case": args.samples, "cases": [],
              "atol": 0.0001, "rtol": 0.0001}
    env = os.environ.copy()
    env["PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;" + env["PATH"]
    cases = [("CLOUD03", .0, .5), ("CLOUD03", .5, 2.0),
             ("CLOUD03", .857, 6.0), ("BUNNY", .5, 1.0)]
    with tempfile.TemporaryDirectory(prefix="mrpnn-trace-") as temp:
        for scene, g, alpha in cases:
            path = Path(temp) / "trace.bin"
            process = subprocess.run([
                str(root / "build-5080/Release/NetworkProbe.exe"),
                str(root / f"TestCase/{scene}"), str(path), str(args.samples),
                str(g), str(alpha)], cwd=root, env=env, capture_output=True, timeout=180)
            if process.returncode:
                raise RuntimeError(process.stdout.decode(errors="replace")[-2000:] +
                                   process.stderr.decode(errors="replace")[-2000:])
            data = read_trace(path)
            case = {"scene": scene, "g": g, "alpha": alpha, "devices": {}}
            for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
                model.to(device)
                inputs = {k: data[k].to(device) for k in
                          ("density", "transmittance", "phase", "g", "gamma", "albedo")}
                with torch.no_grad():
                    pred = model(**inputs, return_trace=True)
                stats = {}
                for name, value in pred.items():
                    actual = value.cpu()
                    target = data[name]
                    error = (actual - target).abs()
                    stats[name] = dict(
                        max_abs=error.max().item(), mean_abs=error.mean().item(),
                        reference_max=target.abs().max().item(),
                        finite=bool(torch.isfinite(actual).all() and torch.isfinite(target).all()),
                        passed=bool(torch.allclose(actual, target, atol=1e-4, rtol=1e-4)))
                case["devices"][device] = stats
                print(scene, g, alpha, device, json.dumps(stats["radiance"]), flush=True)
            report["cases"].append(case)
        # Verify that the transcription supports backpropagation through every parameter.
        model.cpu()
        model.zero_grad()
        model(**{k: data[k] for k in inputs}).mean().backward()
        report["gradient_check"] = {
            "all_parameters_have_finite_gradients": all(
                p.grad is not None and bool(torch.isfinite(p.grad).all())
                for p in model.parameters()),
            "parameters_with_nonzero_gradients": sum(
                p.grad is not None and bool(p.grad.abs().max() > 0)
                for p in model.parameters()),
        }
    report["passed"] = report["gradient_check"]["all_parameters_have_finite_gradients"] and all(
        s["passed"] and s["finite"] for case in report["cases"]
        for stats in case["devices"].values() for s in stats.values())
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), root / "training/original_weights.pt")
    print("Saved", output, "passed =", report["passed"])
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
