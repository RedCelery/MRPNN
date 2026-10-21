#pragma once
#include "mrpnn_features.cuh"
__device__ inline float Uniform(curandState* rng) {
    return fminf(curand_uniform(rng), .99999994f);
}
__device__ inline float TrainingDensity(float3 p) {
    return fmaxf(0.f, tex3D<float>(_DensityVolume, p.z + .5f, p.y + .5f, p.x + .5f));
}
__device__ inline bool Collision(curandState* rng, float3& p, float3 dir,
                          float alpha, float majorant, int& caps) {
    if (alpha == 0 || majorant == 0) return false;
    const float distance = RayBoxDistance(p, dir);
    float t = 0;
    for (int i = 0; i < 1000000; ++i) {
        t -= log1pf(-Uniform(rng)) / majorant;
        if (t >= distance) return false;
        if (Uniform(rng) < alpha * TrainingDensity(p + dir * t) / majorant) {
            p = p + dir * t;
            return true;
        }
    }
    ++caps;
    return false;
}
__device__ inline float Transmission(curandState* rng, float3 p, float3 dir,
                             float alpha, float majorant, int& caps) {
    if (alpha == 0 || majorant == 0) return 1;
    float t = 0, tr = 1;
    const float distance = RayBoxDistance(p, dir);
    for (int i = 0; i < 1000000; ++i) {
        t -= log1pf(-Uniform(rng)) / majorant;
        if (t >= distance) return tr;
        tr *= fmaxf(0.f, 1 - alpha * TrainingDensity(p + dir * t) / majorant);
        if (tr == 0) return 0;
    }
    ++caps;
    return tr;
}
