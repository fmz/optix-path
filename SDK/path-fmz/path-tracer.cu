#include <optix.h>
#include <cuda/helpers.h>

//#include "sutil/vec_math.h"

#include "path-tracer.h"
#include "random.h"

extern "C" {
__constant__ Params params;


struct OrthonormalBasis {
    __forceinline__ __device__ OrthonormalBasis(const float3& normal) {
        m_normal = normal;

        float x = normal.x, y = normal.y, z = normal.z;

        // Lifted from pbrt
        float sign = z >= 0.f ? 1.f : -1.f;
        float a = -1.f / (sign + z);
        float b = x * y * a;

        m_binormal = {b, sign + y*y, -y};
        m_binormal = normalize(m_binormal);
        m_tangent  = cross(m_binormal, m_normal);
    }

    __forceinline__ __device__ void inverse_transform(float3& p) const {
        p = p.x*m_tangent + p.y*m_binormal + p.z*m_normal;
    }

    float3 m_tangent;
    float3 m_binormal;
    float3 m_normal;
};

// Utility functions

static __forceinline__ __device__ RadiancePRD loadClosesthitRadiancePRD()
{
    RadiancePRD prd = {};

    prd.attenuation.x = __uint_as_float(optixGetPayload_0());
    prd.attenuation.y = __uint_as_float(optixGetPayload_1());
    prd.attenuation.z = __uint_as_float(optixGetPayload_2());
    prd.seed  = optixGetPayload_3();
    prd.depth = optixGetPayload_4();
    return prd;
}

static __forceinline__ __device__ RadiancePRD loadMissRadiancePRD() {
    RadiancePRD prd = {};
    return prd;
}

static __forceinline__ __device__ void storeClosesthitRadiancePRD(RadiancePRD prd) {
    optixSetPayload_0(__float_as_uint(prd.attenuation.x));
    optixSetPayload_1(__float_as_uint(prd.attenuation.y));
    optixSetPayload_2(__float_as_uint(prd.attenuation.z));

    optixSetPayload_3(prd.seed);
    optixSetPayload_4(prd.depth);

    optixSetPayload_5(__float_as_uint(prd.emitted.x));
    optixSetPayload_6(__float_as_uint(prd.emitted.y));
    optixSetPayload_7(__float_as_uint(prd.emitted.z));

    optixSetPayload_8(__float_as_uint(prd.radiance.x));
    optixSetPayload_9(__float_as_uint(prd.radiance.y));
    optixSetPayload_10(__float_as_uint(prd.radiance.z));

    optixSetPayload_11(__float_as_uint(prd.origin.x));
    optixSetPayload_12(__float_as_uint(prd.origin.y));
    optixSetPayload_13(__float_as_uint(prd.origin.z));

    optixSetPayload_14(__float_as_uint(prd.direction.x));
    optixSetPayload_15(__float_as_uint(prd.direction.y));
    optixSetPayload_16(__float_as_uint(prd.direction.z));

    optixSetPayload_17(prd.done);
}

static __forceinline__ __device__ void storeMissRadiancePRD(RadiancePRD prd) {
    optixSetPayload_5(__float_as_uint(prd.emitted.x));
    optixSetPayload_6(__float_as_uint(prd.emitted.y));
    optixSetPayload_7(__float_as_uint(prd.emitted.z));

    optixSetPayload_8(__float_as_uint(prd.radiance.x));
    optixSetPayload_9(__float_as_uint(prd.radiance.y));
    optixSetPayload_10(__float_as_uint(prd.radiance.z));

    optixSetPayload_17(prd.done);
}

static __forceinline__ __device__ void cosine_sample_hemisphere(const float u1, const float u2, float3& p)
{
    // Uniformly sample disk.
    const float r   = sqrtf( u1 );
    const float phi = 2.0f*M_PIf * u2;
    p.x = r * cosf( phi );
    p.y = r * sinf( phi );

    // Project up to hemisphere.
    p.z = sqrtf( fmaxf( 0.0f, 1.0f - p.x*p.x - p.y*p.y ) );
}


// static __forceinline__ __device__ void setPayload(float3 p) {
//     optixSetPayload_0(__float_as_uint(p.x));
//     optixSetPayload_1(__float_as_uint(p.y));
//     optixSetPayload_2(__float_as_uint(p.z));
//}

// static __forceinline__ __device__ void computeRay(uint3 idx, uint3 dim, float3& origin, float3& direction) {
//     const float3 U = params.cam_u;
//     const float3 V = params.cam_v;
//     const float3 W = params.cam_w;

//     const float2 d = 2.f * make_float2(
//         static_cast<float>(idx.x) / static_cast<float>(dim.x),
//         static_cast<float>(idx.y) / static_cast<float>(dim.y)
//         ) - 1.f;

//     origin    = params.cam_eye;
//     direction = normalize(d.x * U + d.y * V + W);
// }

static __forceinline__ __device__ void traceRadiance(
    OptixTraversableHandle handle,
    float3                 ray_o,
    float3                 ray_d,
    float                  tmin,
    float                  tmax,
    RadiancePRD&           prd
) {
    unsigned int u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, u17;

    u0 = __float_as_uint(prd.attenuation.x);
    u1 = __float_as_uint(prd.attenuation.y);
    u2 = __float_as_uint(prd.attenuation.z);
    u3 = prd.seed;
    u4 = prd.depth;

    optixTraverse(
        PAYLOAD_TYPE_RADIANCE,
        handle,
        ray_o,
        ray_d,
        tmin,
        tmax,
        0.0f,                     // rayTime
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_NONE,
        0,                        // SBT offset
        RAY_TYPE_COUNT,           // SBT stride
        0,                        // missSBTIndex
        u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, u17
    );

    optixInvoke(PAYLOAD_TYPE_RADIANCE,
        u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, u17
    );

    prd.attenuation = make_float3(__uint_as_float(u0), __uint_as_float(u1), __uint_as_float(u2));
    prd.seed  = u3;
    prd.depth = u4;

    prd.emitted   = make_float3(__uint_as_float(u5),  __uint_as_float(u6),  __uint_as_float(u7));
    prd.radiance  = make_float3(__uint_as_float(u8),  __uint_as_float(u9),  __uint_as_float(u10));
    prd.origin    = make_float3(__uint_as_float(u11), __uint_as_float(u12), __uint_as_float(u13));
    prd.direction = make_float3(__uint_as_float(u14), __uint_as_float(u15), __uint_as_float(u16));
    prd.done      = u17;
}

// Returns true if ray is occluded, else false
static __forceinline__ __device__ bool traceOcclusion(
    OptixTraversableHandle handle,
    float3                 ray_origin,
    float3                 ray_direction,
    float                  tmin,
    float                  tmax
) {
    // We are only casting probe rays so no shader invocation is needed
    optixTraverse(
        handle,
        ray_origin,
        ray_direction,
        tmin,
        tmax, 0.0f,                // rayTime
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0,                         // SBT offset
        RAY_TYPE_COUNT,            // SBT stride
        0                          // missSBTIndex
    );
    return optixHitObjectIsHit();
}

__global__ void __raygen__path_tracer() {
    const int32_t w   = params.width;
    const int32_t h   = params.height;
    const float3  eye = params.cam_eye;
    const float3  U   = params.cam_u;
    const float3  V   = params.cam_v;
    const float3  W   = params.cam_w;
    const float cont_prob = params.continuation_prob;

    const uint3   idx          = optixGetLaunchIndex();
    const int32_t subframe_idx = params.subframe_index;
    // const uint3 dim = optixGetLaunchDimensions();

    // TODO: use a better RNG
    uint32_t seed = tea<4>(idx.y*w + idx.x, subframe_idx);

    float3 result = {0.f, 0.f, 0.f};
    int32_t spl = params.samples_per_launch;
    for (int32_t i = 0; i < spl; i++) {
        // Map thread id to screen coords, and shoot a ray out (in world coords)

        // Vary the target pixel by 0.5 in each direction when mapping to screen space.
        const float2 subpixel_offset = {rnd(seed), rnd(seed)};
        const float2 dir = 2.f * make_float2(
            (float(idx.x) + subpixel_offset.x) / float(w),
            (float(idx.y) + subpixel_offset.y) / float(h)
        ) - 1.f;

        float3 ray_o = eye;
        float3 ray_d = normalize(dir.x * U + dir.y * V + W);

        RadiancePRD prd;
        prd.attenuation = {1.f, 1.f, 1.f};
        prd.seed        = seed;
        prd.depth       = 0;

        while (true) {
            // Trace ray!
            traceRadiance(
                params.handle,
                ray_o,
                ray_d,
                0.0001f,  // tmin
                1e16f,  // tmax
                prd
            );
            result += prd.emitted;
            result += prd.radiance * prd.attenuation;

            const float p = dot(prd.attenuation, {0.3f, 0.59f, 0.11f});
            const bool done = prd.done || rnd(prd.seed) > p;

            // Russian Roulette
            // const float p = rnd(prd.seed);
            // const bool done = prd.done || (p > cont_prob);
            if (done) {
                break;
            }
            prd.attenuation /= p;

            ray_o = prd.origin;
            ray_d = prd.direction;

            prd.depth++;
        }
    }

    const uint32_t image_idx    = idx.y * w + idx.x;
    float3         accum_color  = result / static_cast<float>(params.samples_per_launch); // TODO: use a filter instead

    if(subframe_idx > 0) {
        const float                 a = 1.0f / static_cast<float>( subframe_idx+1 );
        const float3 accum_color_prev = make_float3(params.accum_buffer[image_idx]);
        accum_color = lerp(accum_color_prev, accum_color, a);
    }
    params.accum_buffer[image_idx] = make_float4(accum_color, 1.0f);
    params.frame_buffer[image_idx] = make_color (accum_color);
}

__global__ void __miss__radiance()
{
    optixSetPayloadTypes(PAYLOAD_TYPE_RADIANCE);

    MissData* rt_data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    RadiancePRD prd   = loadMissRadiancePRD();

    prd.radiance  = make_float3(rt_data->bg_clr);
    prd.emitted   = make_float3(0.f);
    prd.done      = true;

    storeMissRadiancePRD(prd);
}


static __forceinline__ __device__ float3 getNormal(const HitGroupData* rt_data, int32_t vert_idx) {
    const float3 n1  = normalize(rt_data->normals[vert_idx+0]);
    // if the
    const float3 n2  = normalize(rt_data->normals[vert_idx+1]);
    const float3 n3  = normalize(rt_data->normals[vert_idx+2]);

    // Get barycentric coords
    const float2 bc = optixGetTriangleBarycentrics();
    float v = bc.x;
    float w = bc.y;
    float u = 1.f - v - w;

    float3 interpolated_normal = normalize(u * n1 + v * n2 + w * n3);
    return interpolated_normal;
}

__global__ void __closesthit__radiance() {
    optixSetPayloadTypes(PAYLOAD_TYPE_RADIANCE);

    HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();

    const int    prim_idx        = optixGetPrimitiveIndex();
    const float3 ray_dir         = optixGetWorldRayDirection();
    const int    vert_idx_offset = prim_idx*3;

    const float3 n    = getNormal(rt_data, vert_idx_offset);

    const float3 N    = n;//aceforward(n, -ray_dir, n);
    const float3 P    = optixGetWorldRayOrigin() + optixGetRayTmax()*ray_dir;

    RadiancePRD prd = loadClosesthitRadiancePRD();

    if(prd.depth == 0) {
        prd.emitted = rt_data->emission_color;
    } else {
        prd.emitted = make_float3( 0.0f );
    }

    unsigned int seed = prd.seed;
    {
        const float z1 = rnd(seed);
        const float z2 = rnd(seed);

        float3 w_in;
        cosine_sample_hemisphere(z1, z2, w_in);
        OrthonormalBasis onb(N);
        onb.inverse_transform(w_in);
        prd.direction = w_in;
        prd.origin    = P;

        prd.attenuation *= rt_data->diffuse_color;
    }

    const float z1 = rnd(seed);
    const float z2 = rnd(seed);
    prd.seed = seed;

    ParallelogramLight light = params.light;
    const float3 light_pos   = light.corner + light.v1 * z1 + light.v2 * z2;

    // Calculate properties of light sample (for area based pdf)
    const float  Ldist = length(light_pos - P );
    const float3 L     = normalize(light_pos - P );
    const float  nDl   = dot( N, L );
    const float  LnDl  = -dot( light.normal, L );

    float weight = 0.0f;
    if( nDl > 0.0f && LnDl > 0.0f ) {
        const bool occluded =
            traceOcclusion(
                params.handle,
                P,
                L,
                0.0001f,           // tmin
                Ldist - 0.0001f);  // tmax

        if( !occluded ) {
            const float A = length(cross(light.v1, light.v2));
            weight = nDl * LnDl * A / (M_PIf * Ldist * Ldist);
        }
    }

    prd.radiance = light.emission * weight;
    prd.done     = false;

    storeClosesthitRadiancePRD( prd );
}
// __global__ void __miss__test_shader() {
//     MissData* miss_data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
//     setPayload(miss_data->bg_clr);
// }

// __global__ void __closesthit__test_shader() {
//     // Neat stuff! we can do barycentrics here.
//     const float2 barycentrics = optixGetTriangleBarycentrics();
//     setPayload(make_float3(barycentrics, 1.f));
// }

}  // extern "C"

