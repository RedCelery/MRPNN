#include "renderResources.cuh"


__device__ cudaTextureObject_t _HGLut = 0;


__device__ cudaTextureObject_t _DensityVolume = 0;

__device__ cudaTextureObject_t _Mips0 = 0;
__device__ cudaTextureObject_t _Mips1 = 0;
__device__ cudaTextureObject_t _Mips2 = 0;
__device__ cudaTextureObject_t _Mips3 = 0;
__device__ cudaTextureObject_t _Mips4 = 0;
__device__ cudaTextureObject_t _Mips5 = 0;
__device__ cudaTextureObject_t _Mips6 = 0;
__device__ cudaTextureObject_t _Mips7 = 0;
__device__ cudaTextureObject_t _Mips8 = 0;

__device__ cudaTextureObject_t _TR_Mips0 = 0;
__device__ cudaTextureObject_t _TR_Mips1 = 0;
__device__ cudaTextureObject_t _TR_Mips2 = 0;
__device__ cudaTextureObject_t _TR_Mips3 = 0;
__device__ cudaTextureObject_t _TR_Mips4 = 0;
__device__ cudaTextureObject_t _TR_Mips5 = 0;
__device__ cudaTextureObject_t _TR_Mips6 = 0;
__device__ cudaTextureObject_t _TR_Mips7 = 0;

__device__ cudaTextureObject_t _HDRI = 0;


__device__ float MipDensityDynamic(int mip, float3 pos) {
    float3 uv = pos + 0.5;
    switch (mip)
    {
    case 0:
        return tex3D<float>(_Mips0, uv.z, uv.y, uv.x);
    case 1:
        return tex3D<float>(_Mips1, uv.z, uv.y, uv.x);
    case 2:
        return tex3D<float>(_Mips2, uv.z, uv.y, uv.x);
    case 3:
        return tex3D<float>(_Mips3, uv.z, uv.y, uv.x);
    case 4:
        return tex3D<float>(_Mips4, uv.z, uv.y, uv.x);
    case 5:
        return tex3D<float>(_Mips5, uv.z, uv.y, uv.x);
    case 6:
        return tex3D<float>(_Mips6, uv.z, uv.y, uv.x);
    case 7:
        return tex3D<float>(_Mips7, uv.z, uv.y, uv.x);
    case 8:
        return tex3D<float>(_Mips8, uv.z, uv.y, uv.x);
    default:
        return 0;
    }
}

__device__ float MipTrDynamic(int mip, float3 pos) {
    float3 uv = pos + 0.5;
    switch (mip)
    {
    case 0:
        return tex3D<float>(_TR_Mips0, uv.z, uv.y, uv.x);
    case 1:
        return tex3D<float>(_TR_Mips1, uv.z, uv.y, uv.x);
    case 2:
        return tex3D<float>(_TR_Mips2, uv.z, uv.y, uv.x);
    case 3:
        return tex3D<float>(_TR_Mips3, uv.z, uv.y, uv.x);
    case 4:
        return tex3D<float>(_TR_Mips4, uv.z, uv.y, uv.x);
    case 5:
        return tex3D<float>(_TR_Mips5, uv.z, uv.y, uv.x);
    case 6:
        return tex3D<float>(_TR_Mips6, uv.z, uv.y, uv.x);
    case 7:
        return tex3D<float>(_TR_Mips7, uv.z, uv.y, uv.x);
    default:
        return 0;
    }
}

__device__ float3 ShadowTerm_TRTex(float3 ori, float3 lightDir, float3 dir, float3 lightColor, float g, int mip)
{
    if (ori.x < -0.5 || ori.y < -0.5 || ori.z < -0.5 || ori.x > 0.5 || ori.y > 0.5 || ori.z > 0.5)
    {
        float offset = RayBoxOffset(ori, lightDir);
        if (offset >= 0)
        {
            return lightColor * MipTr(mip, ori + lightDir * offset);
        }
        else
        {
            return lightColor * float3{ 1.0f,1.0f,1.0f };
        }
    }
    else
    {
        return lightColor * MipTr(mip, ori);
    }
}