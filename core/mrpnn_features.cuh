#pragma once
#include "radiancePredict.cuh"

// Renderer and supervised data generation must use exactly the same descriptor.
__device__ void ExtractMRPNNFeatures(bool active, float3 pos, float3 LightDir,
    float3 XMain, float3 YMain, float3 ZMain, float3 LXMain,
    float alpha, float g, float* density, float* transmittance, float* phase);
