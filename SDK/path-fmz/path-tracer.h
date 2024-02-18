#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#include <optix_types.h>

struct TracerSettings {
    int32_t samplesPerPixel;
    bool directLightingOnly;      // if true, ignore indirect lighting
    int32_t numDirectLightingSamples; // number of shadow rays to trace from each intersection point
    float pathContinuationProb;   // probability of spawning a new secondary ray == (1-pathTerminationProb)
};

constexpr uint32_t RAY_TYPE_COUNT = 1;
constexpr OptixPayloadTypeID PAYLOAD_TYPE_RADIANCE  = OPTIX_PAYLOAD_TYPE_ID_0;

struct RadiancePRD {
    // these are produced by the caller, passed into trace, consumed/modified by CH and MS and consumed again by the caller after trace returned.
    float3   brdf_prod;
    uint32_t seed;
    int32_t  depth;

    // these are produced by CH and MS, and consumed by the caller after trace returned.
    float3   emitted;
    float3   radiance;
    float3   origin;
    float3   direction;
    bool     direct_light_only;
    int32_t  done;
};

const uint32_t radiancePayloadSemantics[18] = {
    // RadiancePRD::brdf_prod
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE,
    // RadiancePRD::seed
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE,
    // RadiancePRD::depth
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE,
    // RadiancePRD::emitted
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
    // RadiancePRD::radiance
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
    // RadiancePRD::origin
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
    // RadiancePRD::direction
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
    // RadiancePRD::done
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE
};

struct ParallelogramLight {
    float3 corner;
    float3 v1, v2;
    float3 normal;
    float3 emission;
};

struct Params {
    uint32_t subframe_index;
    float4*  accum_buffer;
    uchar4*  frame_buffer;
    uint32_t width;
    uint32_t height;
    uint32_t samples_per_launch;
    float continuation_prob;

    float3 cam_eye;
    float3 cam_u, cam_v, cam_w;

    ParallelogramLight   light;  // Hack: we assume a single parallelogram light for this project
    OptixTraversableHandle handle;
};

struct RayGenData {
};

struct MissData {
    float4 bg_clr;
};

enum {
    PHONG_MODEL,
    MIRROR_MODEL,
    TRANSP_MODEL
};

struct MaterialInfo {
    int32_t model;
    float3 diffuse_color;
    float3 emission_color;
    float3 specular_color;
    float  specular_n;
    float  ior;
};

struct HitGroupData {
    float4* vertices;
    float3* normals;
    MaterialInfo mat_info;
};
