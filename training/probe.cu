#include "volume.hpp"
#include "radiancePredict.cuh"
#include "network_trace.cuh"
#include "cuda_objects.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>

extern __device__ NetworkTrace *mrpnnTrace;
extern __device__ int mrpnnTraceCount;

struct ReplaySample {
    float3 pos, view, light, y, z;
    float alpha, g;
    float3 albedo;
};
static_assert(sizeof(ReplaySample) == 20*sizeof(float), "Replay layout");

__global__ void Replay(const ReplaySample* samples, int count) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const auto s = samples[min(id, count-1)];
    const auto lightY = normalize(cross(s.light, float3{0,1,0}));
    curandState seed;
    curand_init(1234, id, 0, &seed);
    RadiancePredict(&seed, id < count, s.pos, s.light, s.view, s.y, s.z,
                    s.light, lightY, cross(s.light, lightY), s.alpha, s.g, s.albedo);
}

__global__ void Probe(int count, float g, float alpha) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    // All lanes participate in the warp shuffle network, including padding.
    const float t = id * 0.61803398875f;
    const float3 pos = {
        .9f * (t - floorf(t)) - .45f,
        .9f * (t * .731f - floorf(t * .731f)) - .45f,
        .9f * (t * .391f - floorf(t * .391f)) - .45f
    };
    const auto view = normalize(float3{ cosf(t), .3f + sinf(t * .7f), sinf(t) });
    const auto viewY = normalize(cross(view, float3{ 0,1,0 }));
    const auto viewZ = cross(view, viewY);
    const auto light = normalize(float3{ .34281f,.70711f,.61845f });
    const auto lightY = normalize(cross(light, float3{ 0,1,0 }));
    const auto lightZ = cross(light, lightY);
    curandState seed;
    curand_init(1234, id, 0, &seed);
    // Distinct channels exercise the three shared albedo heads separately.
    RadiancePredict(&seed, id < count, pos, light, view, viewY, viewZ,
                    light, lightY, lightZ, alpha, g, { .8f,.9f,.999f });
}

int main(int argc, char **argv) {
    try {
        if (argc != 6 && argc != 7) {
            std::cerr << "NetworkProbe volume_without_bin output count g alpha [replay.bin]\n";
            return 1;
        }
        const int count = std::stoi(argv[3]);
        if (count <= 0) throw std::runtime_error("count must be positive");
        const float g = std::stof(argv[4]), alpha = std::stof(argv[5]);
        VolumeRender volume{ std::string(argv[1]) };
        volume.UpdateHGLut(g);
        std::vector<ReplaySample> replay;
        if (argc == 7) {
            std::ifstream input(argv[6], std::ios::binary);
            int n = 0;
            input.read(reinterpret_cast<char*>(&n), sizeof(n));
            if (n != count) throw std::runtime_error("Replay count mismatch");
            replay.resize(count);
            input.read(reinterpret_cast<char*>(replay.data()), count*sizeof(ReplaySample));
            if (!input) throw std::runtime_error("Invalid replay");
            for (auto s : replay)
                if (s.g != g || s.alpha != alpha ||
                    s.light.x != replay[0].light.x || s.light.y != replay[0].light.y ||
                    s.light.z != replay[0].light.z)
                    throw std::runtime_error("Replay requires one g/alpha/light batch");
        }
        volume.Update_TR(replay.empty() ? normalize(float3{ .34281f,.70711f,.61845f }) :
                         replay[0].light, alpha);
        NetworkTrace *device = nullptr;
        CheckCudaObject(cudaMalloc(&device, count * sizeof(NetworkTrace)));
        CheckCudaObject(cudaMemset(device, 0, count * sizeof(NetworkTrace)));
        CheckCudaObject(cudaMemcpyToSymbol(mrpnnTrace, &device, sizeof(device)));
        CheckCudaObject(cudaMemcpyToSymbol(mrpnnTraceCount, &count, sizeof(count)));
        ReplaySample* deviceReplay = nullptr;
        if (replay.empty()) Probe<<<(count + 31) / 32, 32>>>(count, g, alpha);
        else {
            CheckCudaObject(cudaMalloc(&deviceReplay, count*sizeof(ReplaySample)));
            CheckCudaObject(cudaMemcpy(deviceReplay, replay.data(), count*sizeof(ReplaySample),
                                       cudaMemcpyHostToDevice));
            Replay<<<(count+31)/32,32>>>(deviceReplay,count);
        }
        CheckCudaObject(cudaGetLastError());
        CheckCudaObject(cudaDeviceSynchronize());
        std::vector<NetworkTrace> host(count);
        CheckCudaObject(cudaMemcpy(host.data(), device, count * sizeof(NetworkTrace), cudaMemcpyDeviceToHost));
        std::ofstream out{ argv[2], std::ios::binary };
        out.write(reinterpret_cast<const char *>(&count), sizeof(count));
        out.write(reinterpret_cast<const char *>(host.data()), count * sizeof(NetworkTrace));
        if (!out) throw std::runtime_error("Unable to write trace");
        CheckCudaObject(cudaFree(device));
        if (deviceReplay) CheckCudaObject(cudaFree(deviceReplay));
        std::cout << "Exported " << count << " real-volume samples, stride "
                  << sizeof(NetworkTrace) << " bytes\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
