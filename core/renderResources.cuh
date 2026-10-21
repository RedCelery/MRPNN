#pragma once


#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <cuda_runtime.h>
#include <math_functions.h>
#include <device_functions.h>
#include <curand_kernel.h>
#include <cuda_texture_types.h>
#include <cuda_texture_types.h>

#include <vector_types.h>
#include <algorithm>

#include "vector.cuh"

extern __device__ cudaTextureObject_t _HGLut;


extern __device__ cudaTextureObject_t _DensityVolume;

#define Mip(i) _Mips##i
extern __device__ cudaTextureObject_t _Mips0;
extern __device__ cudaTextureObject_t _Mips1;
extern __device__ cudaTextureObject_t _Mips2;
extern __device__ cudaTextureObject_t _Mips3;
extern __device__ cudaTextureObject_t _Mips4;
extern __device__ cudaTextureObject_t _Mips5;
extern __device__ cudaTextureObject_t _Mips6;
extern __device__ cudaTextureObject_t _Mips7;
extern __device__ cudaTextureObject_t _Mips8;
#define TR_Mip(i) _TR_Mips##i
extern __device__ cudaTextureObject_t _TR_Mips0;
extern __device__ cudaTextureObject_t _TR_Mips1;
extern __device__ cudaTextureObject_t _TR_Mips2;
extern __device__ cudaTextureObject_t _TR_Mips3;
extern __device__ cudaTextureObject_t _TR_Mips4;
extern __device__ cudaTextureObject_t _TR_Mips5;
extern __device__ cudaTextureObject_t _TR_Mips6;
extern __device__ cudaTextureObject_t _TR_Mips7;

extern __device__ cudaTextureObject_t _HDRI;

#define MipDensityStatic(mip, pos) tex3D<float>(Mip(mip), (pos).z + 0.5, (pos).y + 0.5, (pos).x + 0.5)
#define MipTrStatic(mip, pos) tex3D<float>(TR_Mip(mip), (pos).z + 0.5, (pos).y + 0.5, (pos).x + 0.5)

__device__ float MipDensityDynamic(int mip, float3 pos);
__device__ float MipTrDynamic(int mip, float3 pos);

//#define MipDensity MipDensityStatic
#define MipDensity MipDensityDynamic
#define MipTr MipTrDynamic


__device__ float3 ShadowTerm_TRTex(float3 ori, float3 lightDir, float3 dir, float3 lightColor, float g, int mip);