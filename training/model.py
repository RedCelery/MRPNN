"""Differentiable transcription of the published MRPNN CUDA forward pass."""
from pathlib import Path
import re
import torch
from torch import nn
from torch.nn import functional as F


def read_weights(path):
    text = Path(path).read_text()
    arrays = re.findall(r"float\s+(\w+)\[(\d+)\]\s*=\s*\{([^}]+)\}", text)
    result = {}
    for name, size, body in arrays:
        values = [float(v.strip().rstrip("f")) for v in body.split(",") if v.strip()]
        if len(values) != int(size):
            raise ValueError(f"{name}: unexpected array size")
        result[name] = torch.tensor(values, dtype=torch.float32)
    if len(result) != 146:
        raise ValueError("Unexpected weight header")
    return result


class MRPNN(nn.Module):
    def __init__(self, weight_header=None):
        super().__init__()
        header = weight_header or Path(__file__).parents[1] / "core/NNWeight.cuh"
        self.weights = nn.ParameterDict({
            name: nn.Parameter(value) for name, value in read_weights(header).items()
        })

    def linear(self, x, name, activation="relu", bias=True):
        w = self.weights[name + "W"].reshape(-1, x.shape[-1])
        b = self.weights[name + "B"] if bias else None
        y = F.linear(x, w, b)
        return F.relu(y) if activation == "relu" else (
            torch.sigmoid(y) if activation == "sigmoid" else y
        )

    def forward(self, density, transmittance, phase, g, gamma, albedo,
                return_trace=False):
        """Features [B,192], g/gamma [B,1], effective albedo [B,3].

        Albedo is the value received by RadiancePredict, after the renderer's
        /1.001 adjustment. Input feature preprocessing already happened in CUDA.
        """
        inputs = (density, transmittance, phase)
        blocks = [
            (0, 0, 8, None, False), (8, 8, 8, 0, True),
            (16, 16, 16, 0, False), (32, 32, 16, 16, False),
            (48, 48, 16, 32, True), (64, 64, 32, 32, False),
            (96, 96, 32, 64, False), (128, 128, 32, 96, False),
        ]
        latent = []
        all_means, all_maxima = [], []
        for high in (False, True):
            states = [{}, {}, {}]
            specs = ([(160 + i * 8, i * 8, 8, None if i == 0 else (i-1)*8, False)
                      for i in range(4)] if high else blocks)
            prefix = "LD" if high else "L"
            for i, (source, dest, size, previous, replace) in enumerate(specs):
                raw = [x[:, source:source+size] for x in inputs]
                means = torch.stack([x.mean(-1) for x in raw], -1)
                maxima = torch.stack([x.max(-1).values for x in raw], -1)
                all_means.append(means)
                all_maxima.append(maxima)
                e = torch.cat((means, maxima, g, gamma), -1)
                se = self.linear(e, f"{prefix}SE{i}1", bias=False)
                gate = self.linear(se, f"{prefix}SE{i}2", "sigmoid", False)
                for c, suffix in enumerate(("", "_Tr", "_Hg")):
                    x = raw[c] * gate[:, c:c+1]
                    if previous is not None:
                        x = x + states[c][previous]
                    y = self.linear(x, f"{prefix}{suffix}{i}1")
                    if replace:
                        # REP3 moves the transformed half back, preserving raw
                        # features in the other half for the next wider block.
                        states[c][previous] = y
                        states[c][dest] = raw[c]
                    else:
                        states[c][dest] = y + raw[c]
                    # Wider blocks read two adjacent halves after REP3.
                    if replace:
                        states[c][previous] = torch.cat((y, raw[c]), -1)
            last = 24 if high else 128
            last_layer = 4 if high else 8
            latent.extend(self.linear(states[c][last], f"{prefix}{suffix}{last_layer}1")
                          for c, suffix in enumerate(("", "_Tr", "_Hg")))
        combined = torch.cat(latent, -1)
        pooled_base = torch.cat(all_means + all_maxima + [g, gamma], -1)
        power = albedo.pow(4)
        heads, gates, reduced, residuals, pools = [], [], [], [], []
        for c in range(3):
            p = power[:, c:c+1]
            pooled = torch.cat((pooled_base, p), -1)
            gate = self.linear(
                self.linear(pooled, "LSEFin1", bias=False),
                "LSEFin2", "sigmoid", False)
            scaled = torch.cat([x * gate[:, i:i+1] for i, x in enumerate(latent)], -1)
            params = self.linear(torch.cat((g, gamma, p), -1), "LGGS")
            x = torch.cat((scaled, params), -1)
            for name in ("LC0", "LC1", "LC2", "LX"):
                x = self.linear(x, name)
            reduced.append(x)
            steps = []
            for i in range(0, 12, 2):
                x = x + self.linear(self.linear(x, f"LX{i}"), f"LX{i+1}")
                steps.append(x)
            heads.append(self.linear(x, "LX12"))
            gates.append(gate)
            residuals.append(torch.cat(steps, -1))
            pools.append(pooled)
        logits = torch.cat(heads, -1)
        # CUDA uses exp(X)-1 (not a fused expm1); retain the same expression.
        radiance = (torch.exp(logits) - 1).clamp_min(0) * power
        if not return_trace:
            return radiance
        return dict(combined=combined, pooled=pools[-1],
                    gates=torch.cat(gates, -1), reduced=torch.cat(reduced, -1),
                    residuals=torch.cat(residuals, -1), logits=logits,
                    radiance=radiance)
