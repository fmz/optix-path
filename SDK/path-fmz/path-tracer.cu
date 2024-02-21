#include <cfloat>
#include <optix.h>
#include <cuda/helpers.h>

//#include "sutil/vec_math.h"

#include "path-tracer.h"
#include "random.h"

extern "C" {
__constant__ Params params;

static __forceinline__ __device__ void _printFloat3(float3 vec, const char* name) {
    printf("%s = %f, %f, %f\n", name, vec.x, vec.y, vec.z);
}

#define printFloat3(x) _printFloat3((x), (#x))

static constexpr float3 zerof3 = {0.f, 0.f, 0.f};

// Quaternion lifted from: https://forums.developer.nvidia.com/t/cuda-for-quaternions-hyper-complex-numbers-operations/44116/2
// Quaternion helper class describing rotations
// this allows for a nice description and execution of rotations in 3D space.
typedef struct _Quaternion {
    // rotation around a given axis (given sine and cosine of HALF the rotation angle)
    static __device__ __forceinline__ struct _Quaternion describe_rotation(
        const float3 v,
        const float sina_2,
        const float cosa_2
    ) {
        struct _Quaternion result;
        result.q = make_float4(cosa_2, sina_2*v.x, sina_2*v.y, sina_2*v.z);
        return result;
    }

    // rotation around a given axis (angle without range restriction)
    static __device__ __forceinline__ struct _Quaternion describe_rotation(
        const float3 v,
        const float angle
    ) {
        float sina_2, cosa_2;
        __sincosf(angle/2.f, &sina_2, &cosa_2);

        struct _Quaternion result;
        result.q = make_float4(cosa_2, sina_2*v.x, sina_2*v.y, sina_2*v.z);
        return result;
    }

    // rotate a point v in 3D space around the origin using this quaternion
    // see EN Wikipedia on Quaternions and spatial rotation
    __device__ __forceinline__ float3 rotate(const float3 v) const {
        float t2 =   q.x*q.y;
        float t3 =   q.x*q.z;
        float t4 =   q.x*q.w;
        float t5 =  -q.y*q.y;
        float t6 =   q.y*q.z;
        float t7 =   q.y*q.w;
        float t8 =  -q.z*q.z;
        float t9 =   q.z*q.w;
        float t10 = -q.w*q.w;
        return make_float3(
            2.0f*( (t8 + t10)*v.x + (t6 -  t4)*v.y + (t3 + t7)*v.z ) + v.x,
            2.0f*( (t4 +  t6)*v.x + (t5 + t10)*v.y + (t9 - t2)*v.z ) + v.y,
            2.0f*( (t7 -  t3)*v.x + (t2 +  t9)*v.y + (t5 + t8)*v.z ) + v.z
        );
    }

    // rotate a point v in 3D space around a given point p using this quaternion
    __device__ __forceinline__ float3 rotate_around_p(const float3 v, const float3 p) {
        return p + rotate(v - p);
    }

protected:
    // 1,i,j,k
    float4 q;
} Quaternion;


struct OrthonormalBasis {
    __forceinline__ __device__ OrthonormalBasis(const float3& normal) {
        // m_normal = normal;

        // float x = normal.x, y = normal.y, z = normal.z;

        // // Lifted from pbrt
        // float sign = z >= 0.f ? 1.f : -1.f;
        // float a = -1.f / (sign + z);
        // float b = x * y * a;

        // m_binormal = {b, sign + y*y, -y};
        // m_binormal = normalize(m_binormal);
        // m_tangent  = cross(m_binormal, m_normal);
        m_normal = normal;

        if(fabs(m_normal.x) > fabs(m_normal.z))
        {
            m_binormal.x = -m_normal.y;
            m_binormal.y =  m_normal.x;
            m_binormal.z =  0;
        }
        else
        {
            m_binormal.x =  0;
            m_binormal.y = -m_normal.z;
            m_binormal.z =  m_normal.y;
        }

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
    prd.seed  = optixGetPayload_6();
    prd.depth = optixGetPayload_7();
    return prd;
}

static __forceinline__ __device__ RadiancePRD loadMissRadiancePRD() {
    RadiancePRD prd = {};
    return prd;
}

static __forceinline__ __device__ void storeClosesthitRadiancePRD(RadiancePRD prd) {
    optixSetPayload_0(__float_as_uint(prd.cur_brdf.x));
    optixSetPayload_1(__float_as_uint(prd.cur_brdf.y));
    optixSetPayload_2(__float_as_uint(prd.cur_brdf.z));
    optixSetPayload_3(__float_as_uint(prd.next_ray_prod.x));
    optixSetPayload_4(__float_as_uint(prd.next_ray_prod.y));
    optixSetPayload_5(__float_as_uint(prd.next_ray_prod.z));

    optixSetPayload_6(prd.seed);
    optixSetPayload_7(prd.depth);

    optixSetPayload_8(__float_as_uint(prd.emitted.x));
    optixSetPayload_9(__float_as_uint(prd.emitted.y));
    optixSetPayload_10(__float_as_uint(prd.emitted.z));

    optixSetPayload_11(__float_as_uint(prd.radiance.x));
    optixSetPayload_12(__float_as_uint(prd.radiance.y));
    optixSetPayload_13(__float_as_uint(prd.radiance.z));

    optixSetPayload_14(__float_as_uint(prd.origin.x));
    optixSetPayload_15(__float_as_uint(prd.origin.y));
    optixSetPayload_16(__float_as_uint(prd.origin.z));

    optixSetPayload_17(__float_as_uint(prd.direction.x));
    optixSetPayload_18(__float_as_uint(prd.direction.y));
    optixSetPayload_19(__float_as_uint(prd.direction.z));

    optixSetPayload_20(prd.direct_light_only);

    optixSetPayload_21(prd.done);
}

static __forceinline__ __device__ void storeMissRadiancePRD(RadiancePRD prd) {
    optixSetPayload_0(__float_as_uint(prd.cur_brdf.x));
    optixSetPayload_1(__float_as_uint(prd.cur_brdf.y));
    optixSetPayload_2(__float_as_uint(prd.cur_brdf.z));

    optixSetPayload_3(__float_as_uint(prd.next_ray_prod.x));
    optixSetPayload_4(__float_as_uint(prd.next_ray_prod.y));
    optixSetPayload_5(__float_as_uint(prd.next_ray_prod.z));

    optixSetPayload_8(__float_as_uint(prd.emitted.x));
    optixSetPayload_9(__float_as_uint(prd.emitted.y));
    optixSetPayload_10(__float_as_uint(prd.emitted.z));

    optixSetPayload_11(__float_as_uint(prd.radiance.x));
    optixSetPayload_12(__float_as_uint(prd.radiance.y));
    optixSetPayload_13(__float_as_uint(prd.radiance.z));

    optixSetPayload_21(prd.done);
}

static __forceinline__ __device__ std::pair<float3, float> sample_from_hemisphere(
    const float rnd0,
    const float rnd1,
    const OrthonormalBasis& onb,
    float3& dir,
    float& inv_pdf
) {
    // Uniformly sample hemisphere
    const float theta  = std::acos(rnd0);
    const float phi    = rnd1 * 2.0f*M_PIf;

    _Quaternion theta_rot = _Quaternion::describe_rotation(onb.m_binormal, theta);
    _Quaternion phi_rot   = _Quaternion::describe_rotation(onb.m_normal, phi);

    dir = phi_rot.rotate_around_p(
        theta_rot.rotate_around_p(onb.m_normal, onb.m_binormal), onb.m_normal);
    inv_pdf = 2.f*M_PIf;
}

static __forceinline__ __device__ void cosine_sample_from_hemisphere(
    const float rnd0,
    const float rnd1,
    const OrthonormalBasis& onb,
    float3& dir,
    float& inv_pdf
) {
    // Sample hemisphere with cosine-theta importance
    const float theta  = std::asin(std::sqrt(rnd0));
    const float phi    = rnd1 * 2.0f*M_PIf;

    _Quaternion theta_rot = _Quaternion::describe_rotation(onb.m_binormal, theta);
    _Quaternion phi_rot   = _Quaternion::describe_rotation(onb.m_normal, phi);

    dir = phi_rot.rotate_around_p(
        theta_rot.rotate_around_p(onb.m_normal, onb.m_binormal), onb.m_normal);
    inv_pdf = M_PIf / std::cos(theta + 0.0001);
}

static __forceinline__ __device__ float3 sample_from_sphere(
    const float rnd0, const float rnd1, const OrthonormalBasis& onb
) {
    // Uniformly sample disk.
    //const float theta  = rnd0 * M_PI_2f;
    const float theta  = std::acos(1.f-(rnd0 * 2.f));
    const float phi    = rnd1 * 2.0f*M_PIf;

    _Quaternion theta_rot = _Quaternion::describe_rotation(onb.m_binormal, theta);
    _Quaternion phi_rot   = _Quaternion::describe_rotation(onb.m_normal, phi);

    return phi_rot.rotate_around_p(theta_rot.rotate_around_p(onb.m_normal, onb.m_binormal), onb.m_normal);
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
    unsigned int u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, u17, u18, u19, u20, u21;

    u6 = prd.seed;
    u7 = prd.depth;

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
        u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, u17, u18, u19, u20, u21
    );

    optixInvoke(PAYLOAD_TYPE_RADIANCE,
        u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, u17, u18, u19, u20, u21
    );

    prd.cur_brdf      = make_float3(__uint_as_float(u0), __uint_as_float(u1), __uint_as_float(u2));
    prd.next_ray_prod = make_float3(__uint_as_float(u3), __uint_as_float(u4), __uint_as_float(u5));
    prd.seed  = u6;
    prd.depth = u7;

    prd.emitted   = make_float3(__uint_as_float(u8),  __uint_as_float(u9),  __uint_as_float(u10));
    prd.radiance  = make_float3(__uint_as_float(u11),  __uint_as_float(u12),  __uint_as_float(u13));
    prd.origin    = make_float3(__uint_as_float(u14), __uint_as_float(u15), __uint_as_float(u16));
    prd.direction = make_float3(__uint_as_float(u17), __uint_as_float(u18), __uint_as_float(u19));
    prd.direct_light_only = u20;
    prd.done      = u21;
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

static __forceinline__ __device__ float getLuminance(const float3 vec) {
    return vec.x * 0.2126f + vec.y * 0.7152f + vec.z * 0.0722f;
}

static __forceinline__ __device__ float3 extendedReindhard(
    const float3 intn,
    const float max_lum
) {
    float max_lum_2 = max_lum * max_lum;
    // Find the output luminance
    float lum_in = getLuminance(intn);
    float lum_out = lum_in * (1.f + (lum_in / max_lum_2));
    lum_out = lum_out / (1.0f + lum_in);
    // Update the luminance
    float3 reinhard = intn * (lum_out / lum_in);

    return reinhard;
}

__global__ void __raygen__path_tracer() {
    const int32_t w   = params.width;
    const int32_t h   = params.height;
    const float3  eye = params.cam_eye;
    const float3  U   = params.cam_u;
    const float3  V   = params.cam_v;
    const float3  W   = params.cam_w;
    const float cont_prob     = params.continuation_prob;
    const float inv_cont_prob = 1.f / cont_prob;


    const uint3   idx          = optixGetLaunchIndex();
    const int32_t subframe_idx = params.subframe_index;
    // const uint3 dim = optixGetLaunchDimensions();

    // TODO: use a better RNG
    uint32_t seed = tea<4>(idx.y*w + idx.x, subframe_idx);

    float3 result = zerof3;
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
        prd.cur_brdf      = zerof3;
        prd.next_ray_prod = zerof3;
        prd.seed       = seed;
        prd.depth      = 0;

        // Primary visibility;
        traceRadiance(
            params.handle,
            ray_o,
            ray_d,
            0.0001f,  // tmin
            1e16f,  // tmax
            prd
        );

        float3 prod_brdf    = prd.cur_brdf;
        float next_ray_prod = prd.next_ray_prod.x;

        float3 direct = prd.emitted + prd.radiance * prod_brdf * next_ray_prod;

        float p = rnd(prd.seed);
        bool done = prd.done || (p > cont_prob);

        direct *= inv_cont_prob;

        ray_o = prd.origin;
        ray_d = prd.direction;

        float3 indirect = zerof3;

        prd.depth++;
        while (!done && prd.depth < 5) {
            traceRadiance(
                params.handle,
                ray_o,
                ray_d,
                0.000001f,  // tmin
                1e16f,  // tmax
                prd
            );

            // p = dot(prod_brdf * next_ray_prod * prd.next_ray_prod * prd.cur_brdf, {0.2126f, 0.7152f, 0.0722f});
            // done = prd.done || rnd(prd.seed) > p;

            // Russian Roulette
            p = rnd(prd.seed);
            done = prd.done || (p > cont_prob);
            if (done) {
                break;
            }

            indirect = indirect
                       + (prd.emitted + prd.radiance) * prod_brdf * next_ray_prod;
            next_ray_prod *= prd.next_ray_prod.x * inv_cont_prob;
            prod_brdf *= prd.cur_brdf;

            ray_o = prd.origin;
            ray_d = prd.direction;

            prd.depth++;
        }
        result += direct + indirect;
    }

    const uint32_t image_idx    = idx.y * w + idx.x;
    float3         accum_color  = result / static_cast<float>(params.samples_per_launch); // TODO: use a filter instead

    if(subframe_idx > 0) {
        const float                 a = 1.0f / static_cast<float>( subframe_idx+1 );
        const float3 accum_color_prev = make_float3(params.accum_buffer[image_idx]);
        accum_color = lerp(accum_color_prev, accum_color, a);
    }

    params.accum_buffer[image_idx] = make_float4(accum_color, 1.0f);


    // Tone map (per-pixel):
    accum_color = extendedReindhard(accum_color, getLuminance(params.light.emission));
    //accum_color = accum_color / (1.f + accum_color);

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
    const float3 n2  = normalize(rt_data->normals[vert_idx+1]);
    const float3 n3  = normalize(rt_data->normals[vert_idx+2]);

    // Get barycentric coords
    const float2 bc = optixGetTriangleBarycentrics();
    float v = bc.x;
    float w = bc.y;
    float u = 1.f - v - w;

    return  normalize(u * n1 + v * n2 + w * n3);
}

// The following functions handle BRDF calculations
// They come in pairs: 1. next-ray sampler (along with its inverse pdf)
//                     2. BRDF, based on the ray sampled from 1.

static __forceinline__ __device__ void diffuse_sample_next_ray(
    const float3 normal,
    uint32_t& seed,
    float3& w_in,
    float&  inv_pdf
) {
    const float z1 = rnd(seed);
    const float z2 = rnd(seed);
    OrthonormalBasis onb(normal);

    cosine_sample_from_hemisphere(z1, z2, onb, w_in, inv_pdf);
}

static __forceinline__ __device__ float3 diffuse_brdf(const MaterialInfo& mat) {
    return mat.diffuse_color / M_PIf;
}

static __forceinline__ __device__ void specular_sample_next_ray(
    const float3 normal,
    const float3 w_out,
    uint32_t& seed,
    float3& w_in,
    float&  inv_pdf
) {
    float3 r = reflect(w_out, normal);

    const float z1 = rnd(seed);
    const float z2 = rnd(seed);
    OrthonormalBasis onb(r);

    //sample_from_hemisphere(z1, z2, onb, w_in, inv_pdf);
    cosine_sample_from_hemisphere(z1, z2, onb, w_in, inv_pdf);
    if (dot(w_in, normal) < 0.f) {
        w_in = -w_in;
    }

}

static __forceinline__ __device__ float3 specular_brdf(
    const MaterialInfo& mat,
    const float3 norm,
    const float3 w_in,
    const float3 w_out
) {
    float3 r = reflect(-w_in, norm);
    float3 ret = {0.f, 0.f, 0.f};
    float r_dot_w = dot(r, w_out);
    float n = mat.specular_n;
    // if (mat.specular_n > 10 && norm.y > 0.99999f) {
    //     printf("\n\n");
    //     printFloat3(w_in);
    //     printFloat3(norm);
    //     printFloat3(w_out);
    //     printFloat3(r);
    //     printFloat3(mat.specular_color);
    //     printf("specular.n = %f, r_dot_w = %f\n\n\n", mat.specular_n, r_dot_w);
    // }
    // if (r_dot_w > 0.9f && r_dot_w < 0.95f) {
    //     // printFloat3(w_in);
    //     // printFloat3(w_out);
    //     // printFloat3(r);
    //     float expon = pow(r_dot_w, n);
    //     printf("r_dot_w = %f, expon = %f\n", r_dot_w, expon);
    // }
    if (r_dot_w > 0.f && n > 0.f) {
        ret = mat.specular_color * ((n + 2.f) / (2.f*M_PIf)) * pow(r_dot_w, n);
    }
    return ret;
}

static __forceinline__ __device__ float3 mirror_brdf(
    const MaterialInfo& mat,
    const float3 norm,
    const float3 w_in,
    const float3 w_out
    ) {

    float3 r = reflect(-w_in, norm);
    float3 ret = {0.f, 0.f, 0.f};
    float r_dot_w = dot(r, w_out);
    // if (mat.specular_n > 100 && norm.y > 0.9999f) {
    //     printFloat3(w_in);
    //     printFloat3(norm);
    //     printFloat3(w_out);
    //     printFloat3(r);
    //     printFloat3(mat.specular_color);
    //     printf("specular.n = %f, r_dot_w = %f\n", mat.specular_n, r_dot_w);
    // }
    //if (fabsf(r_dot_w - 1.f) < 5e-6) {
        ret = mat.specular_color / r_dot_w;
    //}
    return ret;
}

static __forceinline__ __device__ void glass_sample_next_ray(
    const MaterialInfo& mat,
    const float3 normal,
    const float3 w_out,
    uint32_t& seed,
    float3& w_in,
    float&  inv_pdf
) {
    const float z1 = rnd(seed);
    // const float z2 = rnd(seed);
    // OrthonormalBasis onb(normal);

    float wout_dot_n = dot(w_out, normal);
    bool reflect_not_refract;
    bool exiting_ray;
    if (wout_dot_n > 0.f) {
        exiting_ray = false;
        if (z1 > 0.5f) {
            reflect_not_refract = true;
        } else {
            reflect_not_refract = false;
        }
    } else {
        exiting_ray = true;
        if (z1 > 0.5f) {
            reflect_not_refract = false;
        } else {
            reflect_not_refract = true;
        }
    }

    reflect_not_refract = false;

    if (reflect_not_refract) {
        w_in = reflect(w_out, normal);
    } else {
        // Don't trip, w_in is actully the output.
        float ior = exiting_ray ? (1.f/mat.ior) : mat.ior;
        refract(w_in, w_out, -normal, ior);
        // if (fabsf(w_out.y) < 0.000001 ) {
        //     printFloat3(w_in);
        //     printFloat3(w_out);
        // }
    }
    inv_pdf = 2.f;
}

static __forceinline__ __device__ float3 glass_brdf(
    const MaterialInfo& mat,
    const float3 norm,
    const float3 w_in,
    const float3 w_out
 ) {

    float3 refl = reflect(-w_in, norm);
    float3 refr;
    refract(refr, w_in, -norm, mat.ior);

    if (dot(refr, refr) < 1e-6f) {
        // FIXME
        return mat.specular_color;

    }

    // Index of refraction for the incoming ray and for the outgoing ray
    // Remember, we go from out to in, because graphics is backwards
    float n_in, n_out;
    //float norm_dot_wout = dot(norm, w_out);
    bool back_hit = optixIsBackFaceHit();

    if (back_hit) {
        n_in  = 1.f;
        n_out = mat.ior;
    } else {
        // W_out outside, w_in is inside
        n_in  = mat.ior;
        n_out = 1.f;
    }

    // Calculate Schlick's approximation
    float cos_theta_i = dot(w_in, norm);
    float R0 = pow(n_in - n_out, 2) / (n_in + n_out);
    float schlick = R0 - (1.f - R0) * pow(1.f - cos_theta_i, 5);

    float3 ret = {0.f, 0.f, 0.f};
    float refl_dot_w = dot(refl, w_out);
    if (fabsf(w_out.y) < 0.000001 ) {
        printFloat3(w_in);
        printFloat3(norm);
        printFloat3(w_out);
        printFloat3(refl);
        printFloat3(refr);
    }
    float3 specular = {0.f, 0.f, 0.f};
    if (fabsf(refl_dot_w - 1.f) < 1e-6f) {
        specular = mat.specular_color;
    }

    float3 refractive = {0.f, 0.f, 0.f};
    if (fabsf(dot(w_out, refr)) < 1e-6f) {
        refractive += 1.f - mat.specular_color;
    }

    ret = schlick * refractive + (1.f - schlick) * specular;
    return ret;
}

// static __forceinline__ __device__ float3 refractive_brdf(
//     const MaterialInfo& mat,
//     const float3 norm,
//     const float3 w_in,
//     const float3 w_out
//     ) {

//     float3 r = reflect(-w_in, norm);
//     float3 ret = {0.f, 0.f, 0.f};
//     float r_dot_w = dot(r, w_out);
//     // if (mat.specular_n > 100 && norm.y > 0.9999f) {
//     //     printFloat3(w_in);
//     //     printFloat3(norm);
//     //     printFloat3(w_out);
//     //     printFloat3(r);
//     //     printFloat3(mat.specular_color);
//     //     printf("specular.n = %f, r_dot_w = %f\n", mat.specular_n, r_dot_w);
//     // }
//     if (r_dot_w > 0.f && mat.specular_n > 0.f) {
//         ret = mat.specular_color * ((mat.specular_n + 2.f) / (2.f*M_PIf)) * pow(r_dot_w, mat.specular_n);
//     }
//     return ret;
// }

__global__ void __closesthit__radiance() {
    optixSetPayloadTypes(PAYLOAD_TYPE_RADIANCE);

    HitGroupData* rt_data   = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    const MaterialInfo& mat = rt_data->mat_info;

    const int    prim_idx        = optixGetPrimitiveIndex();
    const float3 ray_dir         = normalize(optixGetWorldRayDirection());
    const int    vert_idx_offset = prim_idx*3;

    const float3 n    = getNormal(rt_data, vert_idx_offset);

    const float3 N     = faceforward(n, -ray_dir, n);
    const float3 hit_p = optixGetWorldRayOrigin() + optixGetRayTmax() * ray_dir;
    RadiancePRD prd = loadClosesthitRadiancePRD();

    bool count_emitted = prd.depth == 0 || prd.direct_light_only;  // TODO: rename

    if(count_emitted) {
        prd.emitted = mat.emission_color;
    } else {
        prd.emitted = make_float3( 0.0f );
    }

    //float brdf_normalization = 1.f;
    float3 brdf_blend = zerof3;
    unsigned int seed = prd.seed;
    {
        float3 w_in;
        float inv_pdf;

        switch(mat.model) {
        case PHONG_MODEL:
        {
            if (mat.specular_n > 0.f && (mat.specular_color.x > 0.f || mat.specular_color.y > 0.f || mat.specular_color.z > 0.f)) {
                specular_sample_next_ray(n, ray_dir, seed, w_in, inv_pdf);
                brdf_blend = specular_brdf(mat, n, w_in, -ray_dir);
                // prd.emitted = mat.emission_color;
            } else {
                diffuse_sample_next_ray(N, seed, w_in, inv_pdf);
                brdf_blend = diffuse_brdf(mat);
            }
            break;
        }
        case MIRROR_MODEL:
            w_in = reflect(ray_dir, N);
            inv_pdf = 1;
            brdf_blend += /*diffuse_brdf(mat) +*/ mirror_brdf(mat, N, w_in, -ray_dir);
            count_emitted = false; // sampled light is already direct
            break;
        // case TRANSP_MODEL:
        //     glass_sample_next_ray(mat, N, ray_dir, seed, w_in, inv_pdf);
        //     brdf_blend += glass_brdf(mat, N, w_in, -ray_dir);
        //     break;
        default:
            printf("INVALID MATERIAL MODEL");
            return;
        }
    case TRANSP_MODEL:

        // // Specular contribution
        // if (mat.specular_n > 0 && (
        //         mat.specular_color.x > 0.f ||
        //         mat.specular_color.y > 0.f ||
        //         mat.specular_color.z > 0.f)
        // ) {
        //     w_in[0] = reflect(ray_dir, N);
        //     pdf = 1;
        //     // onb.inverse_transform(w_in);
        //     brdf_blend = specular_brdf(mat, N, w_in[0], -ray_dir);
        // }

        // // Refractive contribution
        // bool refracted = false;
        // if (mat.ior > 0.f) {
        //     if (refract(w_in[1], ray_dir, N, mat.ior)) {
        //         refracted = true;
        //     }
        // }

        // brdf_blend += diffuse_brdf(mat);

        // int32_t num_rays = 2;
        // if (refracted) {
        //     num_rays += 1;
        // }
        // int32_t ray_choice = int32_t(rnd(seed) * num_rays);
        prd.direction = w_in;
        prd.origin    = hit_p;
        float wi_dot_n = dot(w_in, n);

        // scale by cos_theta and sampling probability.
        prd.cur_brdf = brdf_blend;  // costheta
        prd.next_ray_prod.x = wi_dot_n * inv_pdf;

        //brdf_normalization = M_PIf;
    }

    const float z1 = rnd(seed);
    const float z2 = rnd(seed);

    ParallelogramLight light = params.light;
    const float3 light_pos   = light.corner + light.v1 * z1 + light.v2 * z2;

    // Calculate properties of light sample (for area based pdf)
    const float  dist_to_light  = length(light_pos - hit_p);
    const float3 light_dir      = normalize(light_pos - hit_p);
    const float  norm_dot_light = dot(N, light_dir);

    const float  LnDl  = -dot(light.normal, light_dir);

    // Direct light
    if( norm_dot_light > 0.0f && LnDl > 0.0f ) {
        const bool occluded =
            traceOcclusion(
                params.handle,
                hit_p,
                light_dir,
                0.0001f,         // tmin
                dist_to_light);  // tmax

        if(!occluded) {
            const float A = length(cross(light.v1, light.v2));            
            float3 dl_brdf = zerof3;
            // Calculate BRDF for the direct light direction. Only for Phong materials
            // TODO: refactor
            if (mat.specular_n > 0.f && (mat.specular_color.x > 0.f || mat.specular_color.y > 0.f || mat.specular_color.z > 0.f)) {
                dl_brdf = specular_brdf(mat, n, light_dir, -ray_dir);
                // prd.emitted = mat.emission_color;
            } else {
                dl_brdf = diffuse_brdf(mat);
            }
            prd.radiance = dl_brdf * light.emission * (norm_dot_light * LnDl* A) / (M_PIf * dist_to_light * dist_to_light);
        }
    }

    // Count (or don't count emitted for the next iteration)
    prd.direct_light_only = count_emitted;
    prd.done = false;
    prd.seed = seed;

    storeClosesthitRadiancePRD(prd);
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

