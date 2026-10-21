"""Small physical-supervision fit, FP32 export and checkpoint round trip."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import time
import numpy as np
import torch
from model import MRPNN, read_weights

INPUTS = ("density", "transmittance", "phase", "g", "gamma", "albedo")


def export_header(model, path):
    lines = ["#pragma once", "// Physical smoke fit: FP32, CUDA [out,in] flattened weights."]
    original = (Path(__file__).resolve().parents[1]/"core/NNWeight.cuh").read_text()
    # Preserve the source's constant/global placement (constant memory is 64 KiB).
    storage = {name: declaration for declaration, name in re.findall(
        r"(__device__\s+(?:__constant__|const))\s+float\s+(\w+)", original)}
    for name, parameter in model.weights.items():
        values = parameter.detach().cpu().numpy().reshape(-1)
        assert np.isfinite(values).all()
        # Nine significant digits round-trip every finite float32.
        formatted = []
        for value in values:
            text = format(float(value), ".9g")
            if "." not in text and "e" not in text:
                text += ".0"
            formatted.append(text + "f")
        lines += [f"{storage[name]} float {name}[{len(values)}] = {{",
                  ",".join(formatted), "};"]
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")
    restored = read_weights(path)
    assert all(torch.equal(restored[k], v.detach().cpu()) for k, v in model.weights.items())


def metrics(model, data):
    with torch.no_grad():
        trace = model(**{k: data[k] for k in INPUTS}, return_trace=True)
        pred = trace["radiance"]
        # CUDA terminal head is ReLU, so logits are the deployment compression.
        return {
            "log_mse": float((trace["logits"]-data["target_log"]).square().mean()),
            "physical_rmse": float((pred-data["label"]).square().mean().sqrt()),
            "physical_mae": float((pred-data["label"]).abs().mean()),
            "rmse_per_channel": (pred-data["label"]).square().mean(0).sqrt().cpu().tolist(),
            "mc_rms_standard_error": data["standard_error"].square().mean(0).sqrt().cpu().tolist(),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--output", type=Path, default=Path("training/runs/physical-smoke"))
    args = parser.parse_args()
    assert args.steps > 0 and args.batch > 0 and args.lr > 0
    root = Path(__file__).resolve().parents[1]
    out = root / args.output
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(20261021)
    np.random.seed(20261021)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MRPNN().to(device)
    initial_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    files = ["cloud_isotropic-4096.npz", "cloud_forward-4096.npz", "bunny-4096.npz"]
    generation = json.loads((root/"training/datasets/smoke/generation-report.json").read_text())
    assert generation["passed"]
    pieces = []
    for name in files:
        path = root/"training/datasets/smoke"/name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == next(
            c["sha256"] for c in generation["cases"] if c["file"] == name)
        with np.load(path) as npz:
            assert not npz["path_caps"].any() and npz["target_log_mask"].all()
            pieces.append({k: npz[k].copy() for k in (*INPUTS, "label", "target_log",
                                                     "standard_error", "pos", "view",
                                                     "light", "basis_y", "basis_z", "alpha")})
    # Whole-volume holdout: both cloud batches train; all Bunny points hold out.
    train = {k: torch.from_numpy(np.concatenate([p[k] for p in pieces[:2]])).to(device)
             for k in (*INPUTS, "label", "target_log", "standard_error")}
    holdout = {k: torch.from_numpy(pieces[2][k]).to(device) for k in train}
    report = dict(torch=torch.__version__, device=device, seed=20261021, steps=args.steps,
                  batch=args.batch, learning_rate=args.lr, optimizer="Adam",
                  loss="unweighted MSE on logits vs log1p(physical_label/albedo^4)",
                  initialization="published weights (fine-tuning)",
                  training_samples=2000, holdout_samples=1000,
                  train_files=files[:2], holdout_files=files[2:],
                  initial=dict(train=metrics(model, train), holdout=metrics(model, holdout)),
                  history=[])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start = time.perf_counter()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    gradient_finite = True
    updated = set()
    for step in range(1, args.steps+1):
        idx = torch.randint(len(train["label"]), (args.batch,), device=device)
        optimizer.zero_grad(set_to_none=True)
        trace = model(**{k: train[k][idx] for k in INPUTS}, return_trace=True)
        loss = (trace["logits"]-train["target_log"][idx]).square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            if p.grad.abs().max() > 0:
                updated.add(name)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == args.steps:
            item = dict(step=step, batch_loss=float(loss.detach()),
                        train=metrics(model, train), holdout=metrics(model, holdout))
            report["history"].append(item)
            print(json.dumps(item), flush=True)
    if device == "cuda":
        torch.cuda.synchronize()
    report["training_seconds"] = time.perf_counter()-start
    report["peak_allocated_mib"] = torch.cuda.max_memory_allocated()/2**20 if device == "cuda" else None
    report["final"] = dict(train=metrics(model, train), holdout=metrics(model, holdout))
    model.cpu()
    torch.save(model.state_dict(), out/"weights.pt")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": args.steps, "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else []},
               out/"checkpoint.pt")
    export_header(model, out/"NNWeight.cuh")
    restored = MRPNN()
    restored.load_state_dict(torch.load(out/"weights.pt", weights_only=True))
    # Same CPU execution yields an exact checkpoint round trip.
    small = {k: train[k][:64].cpu() for k in INPUTS}
    with torch.no_grad():
        assert torch.equal(model(**small), restored(**small))
    report["checks"] = dict(
        finite_gradients=gradient_finite, checkpoint_roundtrip=True, header_fp32_roundtrip=True,
        parameters_with_nonzero_gradients=len(updated),
        changed_parameter_arrays=sum(not torch.equal(v, initial_state[k])
                                     for k, v in model.state_dict().items()),
        loss_decreased=report["final"]["train"]["log_mse"] < report["initial"]["train"]["log_mse"],
        physical_train_error_decreased=report["final"]["train"]["physical_rmse"] <
                                      report["initial"]["train"]["physical_rmse"])
    # Replay actual training + holdout sample geometry through CUDA texture sampling.
    replay = np.concatenate([
        np.concatenate([p[k][:32] for k in
                        ("pos", "view", "light", "basis_y", "basis_z", "alpha", "g", "albedo")], 1)
        for p in pieces])
    with open(out/"replay.bin", "wb") as f:
        np.asarray([len(replay)], "<i4").tofile(f)
        replay.astype("<f4").tofile(f)
    report["artifacts_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (out/"NNWeight.cuh", out/"weights.pt", out/"replay.bin")}
    report["passed"] = all(report["checks"][k] for k in (
        "finite_gradients", "checkpoint_roundtrip", "header_fp32_roundtrip",
        "loss_decreased", "physical_train_error_decreased"))
    (out/"fit-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Standalone SVG loss curve using the plotting library, when available.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    points = [dict(step=0, **report["initial"])] + report["history"]
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for name, label in (("train", "Cloud training"), ("holdout", "Bunny volume holdout")):
        ax.plot([p["step"] for p in points], [p[name]["log_mse"] for p in points], label=label)
    ax.set(xlabel="Optimization step", ylabel="Log-domain MSE", yscale="log",
           title="MRPNN physical-supervision smoke fit")
    ax.grid(alpha=.25)
    ax.legend()
    fig.savefig(out/"loss.svg")
    print(json.dumps(report["final"]), "passed =", report["passed"], flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
