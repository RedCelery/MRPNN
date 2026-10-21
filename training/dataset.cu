#include "volume.hpp"
#include "mrpnn_features.cuh"
#include "cuda_objects.hpp"
#include <fstream>
#include <memory>
#include <chrono>
#include <cstdint>

struct Record {
    float density[192], transmittance[192], phase[192];
    float g, gamma, albedo[3], alpha;
    float pos[3], view[3], light[3], basis_y[3], basis_z[3];
    float label[3], variance[3], direct, direct_variance, direct_reference;
    float path_caps, tracking_caps, sample_failures;
};
static_assert(sizeof(Record) == 609 * sizeof(float), "Dataset layout");

__device__ float Uniform(curandState* rng) {
    return fminf(curand_uniform(rng), .99999994f);
}
__device__ float TrainingDensity(float3 p) {
    return fmaxf(0.f, tex3D<float>(_DensityVolume, p.z + .5f, p.y + .5f, p.x + .5f));
}
// Exact delta/ratio tracking for the nonnegative texture density, no Tr threshold.
__device__ bool Collision(curandState* rng, float3& p, float3 dir,
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
__device__ float Transmission(curandState* rng, float3 p, float3 dir,
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
__device__ void Store(float* out, float3 v) {
    out[0] = v.x; out[1] = v.y; out[2] = v.z;
}
__device__ float3 Load(const float* x) { return {x[0], x[1], x[2]}; }

__global__ void Features(Record* records, int count, float g, float alpha,
                         float3 albedo, float3 light, float majorant,
                         unsigned long long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;
    auto& r = records[id];
    curandState rng;
    curand_init(seed, id, 0, &rng);
    float3 p{}, view{};
    bool found = false;
    int caps = 0;
    for (int attempt = 0; attempt < 65536; ++attempt) {
        const float3 origin = UniformSampleSphere({Uniform(&rng), Uniform(&rng)}) * 2.f;
        const float3 target = UniformSampleSphere({Uniform(&rng), Uniform(&rng)}) * .35f;
        view = normalize(target - origin);
        p = origin + view * (RayBoxOffset(origin, view) + 1e-6f);
        if (alpha == 0 || majorant == 0) { p = target; found = true; break; }
        if (Collision(&rng, p, view, alpha, majorant, caps)) { found = true; break; }
    }
    r.sample_failures = found ? 0 : 1;
    r.tracking_caps = float(caps);
    const float3 axis = fabsf(view.y) < .9f ? float3{0,1,0} : float3{1,0,0};
    // Random rotation around view, as the runtime random sampling basis.
    const float3 y0 = normalize(cross(view, axis)), z0 = cross(view, y0);
    const float angle = Uniform(&rng) * 6.283185307f;
    const float3 y = y0 * cosf(angle) + z0 * sinf(angle), z = cross(view, y);
    ExtractMRPNNFeatures(found, p, light, view, y, z, light, alpha, g,
                         r.density, r.transmittance, r.phase);
    r.g = g; r.gamma = acosf(fminf(1.f, fmaxf(-1.f, dot(view, light))));
    r.alpha = alpha;
    Store(r.albedo, albedo); Store(r.pos, p); Store(r.view, view);
    Store(r.light, light); Store(r.basis_y, y); Store(r.basis_z, z);
    // Independent deterministic quadrature reference for the direct-light test.
    float distance = RayBoxDistance(p, light), optical = 0;
    for (int i = 0; i < 4096; ++i)
        optical += TrainingDensity(p + light * (distance * (i + .5f) / 4096));
    r.direct_reference = expf(-alpha * optical * distance / 4096) *
                         HenyeyGreenstein(dot(view, light), g);
}

__global__ void Labels(Record* records, int spp, int max_bounces,
                       float majorant, unsigned long long seed) {
    const int id = blockIdx.x, lane = threadIdx.x;
    auto& r = records[id];
    double sums[8]{};
    int caps = 0, path_caps = 0;
    for (int s = lane; s < spp; s += blockDim.x) {
        curandState rng;
        curand_init(seed, (unsigned long long)id * spp + s, 0, &rng);
        float3 pos = Load(r.pos), view = Load(r.view), light = Load(r.light);
        const float direct = Transmission(&rng, pos, light, r.alpha, majorant, caps) *
                             HenyeyGreenstein(dot(view, light), r.g);
        float3 dir = SampleHenyeyGreenstein(Uniform(&rng), Uniform(&rng), view, r.g);
        float3 throughput{1,1,1}, result{};
        for (int bounce = 0; bounce < max_bounces; ++bounce) {
            if (!Collision(&rng, pos, dir, r.alpha, majorant, caps)) break;
            throughput = throughput * Load(r.albedo) * 1.001f;
            const float contribution = Transmission(&rng, pos, light, r.alpha, majorant, caps) *
                                       HenyeyGreenstein(dot(dir, light), r.g);
            result = result + throughput * contribution;
            if (bounce == max_bounces - 1) { ++path_caps; break; }
            // Unbiased roulette, independent of channel; compensate survivors.
            if (bounce >= 31) {
                const float survival = fminf(.99f, fmaxf(.05f,
                    fmaxf(throughput.x, fmaxf(throughput.y, throughput.z))));
                if (Uniform(&rng) >= survival) break;
                throughput = throughput / survival;
            }
            dir = SampleHenyeyGreenstein(Uniform(&rng), Uniform(&rng), dir, r.g);
        }
        float v[4] = {result.x, result.y, result.z, direct};
        for (int c = 0; c < 4; ++c) {
            sums[c] += v[c]; sums[c+4] += double(v[c]) * v[c];
        }
    }
    __shared__ double reduction[128][8];
    for (int c = 0; c < 8; ++c) reduction[lane][c] = sums[c];
    __syncthreads();
    for (int step = 64; step; step >>= 1) {
        if (lane < step) for (int c = 0; c < 8; ++c)
            reduction[lane][c] += reduction[lane+step][c];
        __syncthreads();
    }
    atomicAdd(&r.path_caps, float(path_caps));
    atomicAdd(&r.tracking_caps, float(caps));
    if (lane == 0) {
        for (int c = 0; c < 4; ++c) {
            float mean = float(reduction[0][c] / spp);
            float variance = float(fmax(0., (reduction[0][c+4] -
                reduction[0][c] * reduction[0][c] / spp) / (spp - 1)));
            if (c < 3) { r.label[c] = mean; r.variance[c] = variance; }
            else { r.direct = mean; r.direct_variance = variance; }
        }
    }
}

int main(int argc, char** argv) {
    try {
        if (argc != 13) throw std::runtime_error(
            "DatasetGen volume|empty|constant output count spp bounces g alpha ax ay az seed light_angle");
        int count = std::stoi(argv[3]), spp = std::stoi(argv[4]), bounces = std::stoi(argv[5]);
        float g = std::stof(argv[6]), alpha = std::stof(argv[7]);
        float3 albedo{std::stof(argv[8]), std::stof(argv[9]), std::stof(argv[10])};
        auto seed = std::stoull(argv[11]);
        float angle = std::stof(argv[12]);
        if (count < 1 || count > 1000000 || spp < 2 || bounces < 1 || bounces > 4096 ||
            !std::isfinite(g) || fabsf(g) >= .99f || !std::isfinite(alpha) || alpha < 0 ||
            !std::isfinite(angle) ||
            !std::isfinite(albedo.x) || !std::isfinite(albedo.y) || !std::isfinite(albedo.z) ||
            albedo.x < 0 || albedo.y < 0 || albedo.z < 0 ||
            albedo.x > 1.f/1.001f || albedo.y > 1.f/1.001f || albedo.z > 1.f/1.001f)
            throw std::runtime_error("Invalid parameters (physical albedo: effective <= 1/1.001)");
        std::unique_ptr<VolumeRender> volume;
        std::string path = argv[1];
        if (path == "empty" || path == "constant") {
            volume = std::make_unique<VolumeRender>(256);
            volume->SetDatas([&](int,int,int,float,float,float) { return path == "empty" ? 0.f : 1.f; });
            volume->Update();
        } else {
            std::ifstream file(path + ".bin", std::ios::binary);
            int resolution = 0;
            file.read(reinterpret_cast<char*>(&resolution), 4);
            file.seekg(0, std::ios::end);
            if (!file || resolution < 2 || resolution > 1024 ||
                file.tellg() != 4 + int64_t(resolution)*resolution*resolution*4)
                throw std::runtime_error("Missing or invalid volume .bin");
            volume = std::make_unique<VolumeRender>(path);
        }
        const float3 light = normalize(float3{cosf(angle), .70711f, sinf(angle)});
        volume->UpdateHGLut(g);
        volume->Update_TR(light, alpha);
        CheckCudaObject(cudaDeviceSynchronize());
        Record* device;
        CheckCudaObject(cudaMalloc(&device, count * sizeof(Record)));
        CheckCudaObject(cudaMemset(device, 0, count * sizeof(Record)));
        const float majorant = path == "empty" ? 0.f : volume->max_density * alpha;
        Features<<<(count+127)/128,128>>>(device,count,g,alpha,albedo,light,majorant,seed);
        CheckCudaObject(cudaGetLastError());
        CheckCudaObject(cudaDeviceSynchronize());
        auto start = std::chrono::steady_clock::now();
        Labels<<<count,128>>>(device,spp,bounces,majorant,seed ^ 0xABCDEF123ULL ^ spp);
        CheckCudaObject(cudaGetLastError());
        CheckCudaObject(cudaDeviceSynchronize());
        const double ms = std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-start).count();
        std::vector<Record> host(count);
        CheckCudaObject(cudaMemcpy(host.data(),device,count*sizeof(Record),cudaMemcpyDeviceToHost));
        CheckCudaObject(cudaFree(device));
        for (auto& r : host) if (r.sample_failures || r.tracking_caps)
            throw std::runtime_error("Sampling/tracking cap reached; dataset rejected");
        std::ofstream out(argv[2], std::ios::binary);
        int header[5] = {0x4D525044,1,count,spp,609};
        out.write(reinterpret_cast<char*>(header),sizeof(header));
        out.write(reinterpret_cast<char*>(host.data()),count*sizeof(Record));
        if (!out) throw std::runtime_error("Cannot write dataset");
        std::cout << "label_ms=" << ms << " count=" << count << " spp=" << spp << '\n';
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
