"""Generate physical MRPNN supervision and run independent estimator checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import numpy as np
import torch
from model import MRPNN

FIELDS = [
    ("density", 192), ("transmittance", 192), ("phase", 192),
    ("g", 1), ("gamma", 1), ("albedo", 3), ("alpha", 1),
    ("pos", 3), ("view", 3), ("light", 3), ("basis_y", 3), ("basis_z", 3),
    ("label", 3), ("variance", 3), ("direct", 1), ("direct_variance", 1),
    ("direct_reference", 1), ("path_caps", 1), ("tracking_caps", 1),
    ("sample_failures", 1),
]


def read_dataset(path, allow_path_caps=False):
    with open(path, "rb") as f:
        magic, version, count, spp, stride = np.fromfile(f, "<i4", 5)
        assert magic == 0x4D525044 and version == 1 and stride == sum(n for _, n in FIELDS)
        raw = np.fromfile(f, "<f4").reshape(count, stride)
    data, offset = {}, 0
    for name, size in FIELDS:
        data[name] = raw[:, offset:offset+size].copy()
        offset += size
    assert np.isfinite(raw).all()
    assert (data["label"] >= 0).all() and (data["variance"] >= 0).all()
    assert not data["tracking_caps"].any() and not data["sample_failures"].any()
    assert allow_path_caps or not data["path_caps"].any(), "Bounce cap reached: increase --bounces"
    power = data["albedo"] ** 4
    data["target_log"] = np.log1p(np.divide(data["label"], power,
                                         out=np.zeros_like(power), where=power > 0))
    data["target_log_mask"] = power > 0
    data["standard_error"] = np.sqrt(data["variance"] / spp)
    return data


def second_order_reference(pos, light, alpha):
    """Independent angular/free-flight quadrature, g=0, homogeneous unit cube.

    Constant texture has a subvoxel border ramp, handled as a tolerance here.
    """
    mu, mw = np.polynomial.legendre.leggauss(48)
    phi = (np.arange(96) + .5) * 2*np.pi/96
    x = np.sqrt(1-mu[:, None]**2) * np.cos(phi)
    y = np.sqrt(1-mu[:, None]**2) * np.sin(phi)
    z = np.broadcast_to(mu[:, None], x.shape)
    dirs = np.stack((x, y, z), -1).reshape(-1, 3)
    weights = np.broadcast_to(mw[:, None]/(2*96), x.shape).ravel()
    nodes, nw = np.polynomial.legendre.leggauss(48)
    results = []
    for p in pos:
        limit = np.min(((np.where(dirs > 0, .5, -.5)-p)/dirs), axis=-1)
        distances = limit[:, None] * (nodes+1)/2
        points = p + dirs[:, None, :] * distances[:, :, None]
        light_limit = np.min((np.where(light > 0, .5, -.5)-points)/light, axis=-1)
        integral = np.sum(alpha * np.exp(-alpha*(distances+light_limit)) *
                          nw * limit[:, None]/2, axis=-1)
        results.append(np.dot(weights, integral)*.25)
    return np.asarray(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--spp", type=int, nargs=2, default=[512, 4096])
    parser.add_argument("--bounces", type=int, default=2048)
    parser.add_argument("--output", type=Path, default=Path("training/datasets/smoke"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root / args.output
    out.mkdir(parents=True, exist_ok=True)
    (out/"generation-report.json").write_text(
        json.dumps({"passed": False, "status": "generation_in_progress"}), encoding="utf-8")
    env = os.environ.copy()
    env["PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;" + env["PATH"]
    report = {"samples_per_real_case": args.samples, "spp": args.spp,
              "max_bounces": args.bounces, "cases": [], "checks": {}}
    report["source_sha256"] = {
        p: hashlib.sha256((root/p).read_bytes()).hexdigest()
        for p in ("core/radiancePredict.cu", "training/dataset.cu",
                  "training/generate_dataset.py", "core/vector.cu")}
    report["volume_sha256"] = {
        p: hashlib.sha256((root/p).read_bytes()).hexdigest()
        for p in ("TestCase/CLOUD03.bin", "TestCase/BUNNY.bin")}
    torch.set_num_threads(2)
    model = MRPNN()
    cases = [
        ("empty", "empty", 0., 1., [.8, .9, .999], 32, 8192, 512, .7),
        ("zero_alpha", "constant", 0., 0., [.8, .9, .999], 32, 8192, 512, .7),
        ("zero_albedo", "constant", 0., 1., [0, 0, 0], 32, 8192, 512, .7),
        ("single_bounce", "constant", 0., 1., [.8, .9, .999], 32, 65536, 1, .7),
        ("cloud_isotropic", "TestCase/CLOUD03", 0., .5, [.8, .9, .999], args.samples, None, args.bounces, .2),
        ("cloud_forward", "TestCase/CLOUD03", .857, 6., [.8, .9, .999], args.samples, None, args.bounces, 1.4),
        ("bunny", "TestCase/BUNNY", .5, 1., [.8, .9, .999], args.samples, None, args.bounces, 2.7),
    ]
    saved = {}
    with tempfile.TemporaryDirectory(prefix="mrpnn-dataset-") as temp:
        for name, scene, g, alpha, albedo, count, fixed_spp, bounces, angle in cases:
            for spp in [fixed_spp] if fixed_spp else args.spp:
                binary = Path(temp) / "dataset.bin"
                seed = 20261021
                command = [str(root/"build-5080/Release/DatasetGen.exe"),
                           scene if scene in ("empty", "constant") else str(root/scene),
                           str(binary), str(count), str(spp), str(bounces), str(g),
                           str(alpha), *map(str, albedo), str(seed), str(angle)]
                start = time.perf_counter()
                process = subprocess.run(command, cwd=root, env=env, capture_output=True, timeout=600)
                log = process.stdout.decode(errors="replace") + process.stderr.decode(errors="replace")
                if process.returncode:
                    raise RuntimeError(log[-4000:])
                data = read_dataset(binary, allow_path_caps=name == "single_bounce")
                # Cap is intentionally active in the second-order-only diagnostic.
                saved[(name, spp)] = data
                inputs = {k: torch.from_numpy(data[k]) for k in
                          ("density", "transmittance", "phase", "g", "gamma", "albedo")}
                with torch.no_grad():
                    pred = model(**inputs).numpy()
                assert np.isfinite(pred).all()
                data["original_prediction"] = pred
                file = out / f"{name}-{spp}.npz"
                np.savez_compressed(file, **data)
                stderr = data["standard_error"]
                direct_error = np.abs(data["direct"]-data["direct_reference"])
                direct_se = np.sqrt(data["direct_variance"]/spp)
                case = dict(name=name, scene=scene, g=g, alpha=alpha, albedo=albedo,
                            seed=seed, light_angle=angle, samples=count, spp=spp, bounces=bounces,
                            label_ms=float(re.search(r"label_ms=([\d.]+)", log)[1]),
                            wall_seconds=time.perf_counter()-start,
                            label_mean=data["label"].mean(axis=0).tolist(),
                            mean_standard_error=stderr.mean(axis=0).tolist(),
                            rms_standard_error=np.sqrt((stderr**2).mean(axis=0)).tolist(),
                            direct_max_abs_error=float(direct_error.max()),
                            direct_check_passed=bool((direct_error <= 6*direct_se + .001).all()),
                            path_caps=int(data["path_caps"].sum()),
                            sha256=hashlib.sha256(file.read_bytes()).hexdigest(), file=file.name)
                report["cases"].append(case)
                print(json.dumps(case), flush=True)
    for name in ("empty", "zero_alpha", "zero_albedo"):
        d = saved[(name, 8192)]
        report["checks"][name] = bool(np.all(d["label"] == 0))
    d = saved[("single_bounce", 65536)]
    ref = second_order_reference(d["pos"], d["light"][0], 1.)[:, None]*d["albedo"]*1.001
    # Explicit border approximation budget: 0.001 absolute radiance.
    report["checks"]["independent_second_order"] = bool(
        (np.abs(d["label"]-ref) <= 6*d["standard_error"]+.001).all())
    report["second_order_max_abs"] = float(np.abs(d["label"]-ref).max())
    report["convergence"] = {}
    for name in ("cloud_isotropic", "cloud_forward", "bunny"):
        low, high = [saved[(name, n)] for n in args.spp]
        for k in ("density", "transmittance", "phase", "pos", "albedo"):
            assert np.array_equal(low[k], high[k]), "SPP must not change sample/features"
        delta = high["label"]-low["label"]
        expected = (low["variance"]/args.spp[0]+high["variance"]/args.spp[1]).mean(axis=0)
        report["convergence"][name] = dict(
            observed_rms_delta=np.sqrt((delta**2).mean(axis=0)).tolist(),
            expected_rms_delta=np.sqrt(expected).tolist(),
            stderr_high_over_low=(np.sqrt((high["standard_error"]**2).mean(axis=0)) /
                                 np.sqrt((low["standard_error"]**2).mean(axis=0))).tolist())
    report["passed"] = all(report["checks"].values()) and all(
        c["direct_check_passed"] and (c["path_caps"] == 0 or c["name"] == "single_bounce")
        for c in report["cases"])
    report["convergence_check_passed"] = all(
        # Empirical variance can be heavy-tailed: require reduced uncertainty,
        # while reporting deviation from the ideal sqrt(low/high) separately.
        max(c["stderr_high_over_low"]) < 1 and
        np.all(np.abs(np.asarray(c["observed_rms_delta"]) /
                      np.asarray(c["expected_rms_delta"])-1) < .3)
        for c in report["convergence"].values())
    report["passed"] = report["passed"] and report["convergence_check_passed"]
    report["ideal_stderr_ratio"] = float(np.sqrt(args.spp[0]/args.spp[1]))
    report["noise_warning"] = (
        "Near-conservative albedo has heavy-tailed paths; high-SPP uncertainty "
        "is lower in this batch but is not uniformly proportional to 1/sqrt(spp).")
    (out/"generation-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("passed =", report["passed"], flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
