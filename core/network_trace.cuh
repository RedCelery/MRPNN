#pragma once
// Float-only layout shared by the CUDA probe and Python reader.
struct NetworkTrace {
    float g, gamma, albedo[3];
    float density[192], transmittance[192], phase[192];
    float combined[120], pooled[75];
    float gates[18], reduced[48], residuals[288];
    float logits[3], radiance[3];
};
