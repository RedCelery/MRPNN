#include "volume.hpp"
#include "tracking.cuh"
#include "cuda_objects.hpp"
#include <fstream>
#include <chrono>

struct Pixel { float3 full, indirect, standard_error; };
static_assert(sizeof(Pixel)==9*sizeof(float), "Pixel layout");
__device__ float3 MCIndirect(curandState* rng, float3 pos, float3 view, float3 light,
    float alpha, float g, float majorant, float3 albedo, unsigned* caps) {
    int tracking = 0;
    float3 dir = SampleHenyeyGreenstein(Uniform(rng), Uniform(rng), view, g);
    float3 throughput{1,1,1}, result{};
    for (int bounce = 0; bounce < 2048; ++bounce) {
        if (!Collision(rng,pos,dir,alpha,majorant,tracking)) break;
        throughput = throughput * albedo * 1.001f;
        result = result + throughput * (Transmission(rng,pos,light,alpha,majorant,tracking) *
                 HenyeyGreenstein(dot(dir,light),g));
        if (bounce == 2047) { atomicAdd(caps,1U); break; }
        if (bounce >= 31) {
            const float survival = fminf(.99f,fmaxf(.05f,
                fmaxf(throughput.x,fmaxf(throughput.y,throughput.z))));
            if (Uniform(rng) >= survival) break;
            throughput = throughput / survival;
        }
        dir = SampleHenyeyGreenstein(Uniform(rng),Uniform(rng),dir,g);
    }
    if (tracking) atomicAdd(caps,unsigned(tracking));
    return result;
}

__global__ void RenderComparison(Pixel* pixels, unsigned* caps, int width, int spp,
    bool reference, float3 camera, float3 light, float alpha, float g,
    float majorant, float3 albedo, unsigned long long seed) {
    const int id = blockIdx.x*blockDim.x+threadIdx.x;
    const int count = width*width;
    const int px = id%width, py = id/width;
    float3 forward = normalize(-camera);
    float3 right = normalize(cross(forward,float3{0,1,0}));
    float3 up = cross(right,forward);
    float3 sum{}, indirect_sum{}, square{};
    for (int s = 0; s < spp; ++s) {
        curandState rng;
        curand_init(seed,(unsigned long long)id*spp+s,0,&rng);
        float u = ((px+Uniform(&rng))/width*2-1)*.42f;
        float v = (1-(py+Uniform(&rng))/width*2)*.42f;
        float3 dir = normalize(forward+right*u+up*v);
        const float offset = RayBoxOffset(camera,dir);
        float3 pos = camera + dir*fmaxf(0.f,offset);
        int tracking = 0;
        bool active = id<count && offset>=0 &&
                      Collision(&rng,pos,dir,alpha,majorant,tracking);
        if (tracking) atomicAdd(caps,unsigned(tracking));
        // All lanes call the network, even inactive samples, for warp shuffles.
        const float3 axis = fabsf(dir.y)<.9f ? float3{0,1,0} : float3{1,0,0};
        float3 y0=normalize(cross(dir,axis)), z0=cross(dir,y0);
        float rotation=Uniform(&rng)*6.283185307f;
        float3 y=y0*cosf(rotation)+z0*sinf(rotation), z=cross(dir,y);
        float3 indirect{};
        if (!reference) {
            float3 ly=normalize(cross(light,float3{0,1,0}));
            indirect=RadiancePredict(&rng,active,pos,light,dir,y,z,
                                     light,ly,cross(light,ly),alpha,g,albedo);
        } else if (active) {
            indirect=MCIndirect(&rng,pos,dir,light,alpha,g,majorant,albedo,caps);
        }
        float3 full{};
        if (active) {
            int tr_caps=0;
            float direct=Transmission(&rng,pos,light,alpha,majorant,tr_caps) *
                         HenyeyGreenstein(dot(dir,light),g);
            if (tr_caps) atomicAdd(caps,unsigned(tr_caps));
            const float3 outer=albedo*1.001f;
            full=(indirect+float3{direct,direct,direct})*outer;
            indirect=indirect*outer;
        } else indirect={0,0,0};
        sum=sum+full;
        indirect_sum=indirect_sum+indirect;
        square=square+full*full;
    }
    if (id<count) {
        pixels[id].full=sum/float(spp);
        pixels[id].indirect=indirect_sum/float(spp);
        float3 variance=max((square-sum*sum/float(spp))/float(spp-1),float3{0,0,0});
        pixels[id].standard_error={
            sqrtf(variance.x/spp),sqrtf(variance.y/spp),sqrtf(variance.z/spp)};
    }
}

int main(int argc,char** argv) {
    try {
        if (argc!=9) throw std::runtime_error(
            "RenderCompare volume output width spp reference g alpha light_angle");
        int width=std::stoi(argv[3]), spp=std::stoi(argv[4]);
        bool reference=std::stoi(argv[5])!=0;
        float g=std::stof(argv[6]), alpha=std::stof(argv[7]), angle=std::stof(argv[8]);
        if (width<32 || width>2048 || spp<2) throw std::runtime_error("Invalid size/spp");
        VolumeRender volume{std::string(argv[1])};
        float3 light=normalize(float3{cosf(angle),.70711f,sinf(angle)});
        volume.UpdateHGLut(g);
        volume.Update_TR(light,alpha);
        CheckCudaObject(cudaDeviceSynchronize());
        Pixel* device;
        unsigned* caps;
        CheckCudaObject(cudaMalloc(&device,width*width*sizeof(Pixel)));
        CheckCudaObject(cudaMalloc(&caps,sizeof(unsigned)));
        CheckCudaObject(cudaMemset(caps,0,sizeof(unsigned)));
        auto start=std::chrono::steady_clock::now();
        RenderComparison<<<(width*width+31)/32,32>>>(device,caps,width,spp,reference,
            {1.05f,.62f,1.35f},light,alpha,g,volume.max_density*alpha,
            {.8f,.9f,.999f},20261022ULL);
        CheckCudaObject(cudaGetLastError());
        CheckCudaObject(cudaDeviceSynchronize());
        double ms=std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-start).count();
        std::vector<Pixel> host(width*width);
        unsigned cap_count;
        CheckCudaObject(cudaMemcpy(host.data(),device,host.size()*sizeof(Pixel),cudaMemcpyDeviceToHost));
        CheckCudaObject(cudaMemcpy(&cap_count,caps,sizeof(unsigned),cudaMemcpyDeviceToHost));
        CheckCudaObject(cudaFree(device));
        CheckCudaObject(cudaFree(caps));
        if (cap_count) throw std::runtime_error("Tracking/bounce cap reached");
        std::ofstream out(argv[2],std::ios::binary);
        int header[4]={0x52434D50,width,spp,9};
        out.write(reinterpret_cast<char*>(header),sizeof(header));
        out.write(reinterpret_cast<char*>(host.data()),host.size()*sizeof(Pixel));
        if (!out) throw std::runtime_error("Cannot write image");
        std::cout<<"render_ms="<<ms<<" caps="<<cap_count<<'\n';
    } catch (const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
