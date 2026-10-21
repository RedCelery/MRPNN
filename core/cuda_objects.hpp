#pragma once
#include <cuda_runtime.h>
#include <stdexcept>

inline void CheckCudaObject(cudaError_t result)
{
    if (result != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(result));
}

template<class Symbol>
inline void BindTextureObject(const Symbol &symbol, cudaArray_t array,
                              cudaTextureAddressMode mode)
{
    cudaTextureObject_t old = 0;
    CheckCudaObject(cudaMemcpyFromSymbol(&old, symbol, sizeof(old)));
    // Previous renders must finish before replacing a texture handle.
    CheckCudaObject(cudaDeviceSynchronize());
    if (old) CheckCudaObject(cudaDestroyTextureObject(old));
    cudaResourceDesc resource{};
    resource.resType = cudaResourceTypeArray;
    resource.res.array.array = array;
    cudaTextureDesc texture{};
    texture.normalizedCoords = 1;
    texture.filterMode = cudaFilterModeLinear;
    texture.readMode = cudaReadModeElementType;
    for (auto &address : texture.addressMode) address = mode;
    cudaTextureObject_t handle = 0;
    CheckCudaObject(cudaCreateTextureObject(&handle, &resource, &texture, nullptr));
    CheckCudaObject(cudaMemcpyToSymbol(symbol, &handle, sizeof(handle)));
}

template<class Symbol>
inline void BindSurfaceObject(const Symbol &symbol, cudaArray_t array)
{
    cudaSurfaceObject_t old = 0;
    CheckCudaObject(cudaMemcpyFromSymbol(&old, symbol, sizeof(old)));
    CheckCudaObject(cudaDeviceSynchronize());
    if (old) CheckCudaObject(cudaDestroySurfaceObject(old));
    cudaResourceDesc resource{};
    resource.resType = cudaResourceTypeArray;
    resource.res.array.array = array;
    cudaSurfaceObject_t handle = 0;
    CheckCudaObject(cudaCreateSurfaceObject(&handle, &resource));
    CheckCudaObject(cudaMemcpyToSymbol(symbol, &handle, sizeof(handle)));
}
