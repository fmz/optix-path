#include "hackyBuildAccel.h"
#include <cstdint>

// CUDA
#include <glad/glad.h>  // Needs to be included before gl_interop
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

// Optix
#include <optix.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include "scene/scene.h"
#include "sutil/Exception.h"

struct ParallelogramHackyLight {
    float3 corner;
    float3 v1, v2;
    float3 normal;
    float3 emission;
};

struct HackyParams {
    uint32_t subframe_index;
    float4*  accum_buffer;
    uchar4*  frame_buffer;
    uint32_t width;
    uint32_t height;
    uint32_t samples_per_launch;

    float3 cam_eye;
    float3 cam_u, cam_v, cam_w;

    ParallelogramHackyLight   light;  // FIXME
    OptixTraversableHandle handle;
};

struct PathTracerHackyState {
    OptixDeviceContext context = 0;

    // Accel state
    OptixTraversableHandle gas_handle          = 0;
    CUdeviceptr            d_gas_output_buffer = 0;
    CUdeviceptr            d_vertices          = 0;
    CUdeviceptr            d_indices           = 0;

    OptixModule                 optix_module = 0;
    OptixPipelineCompileOptions pco          = {};
    OptixPipeline               pipeline     = 0;

    OptixProgramGroup raygen_prog_group   = 0;
    OptixProgramGroup radiance_miss_group = 0;
    OptixProgramGroup radiance_hit_group  = 0;

    CUstream stream = 0;

    HackyParams  params;
    HackyParams* d_params;

    OptixShaderBindingTable sbt = {};
};

static std::array<uint32_t, 32> g_mat_indices = {{
    0, 0,                          // Floor         -- white lambert
    0, 0,                          // Ceiling       -- white lambert
    0, 0,                          // Back wall     -- white lambert
    1, 1,                          // Right wall    -- green lambert
    2, 2,                          // Left wall     -- red lambert
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // Short block   -- white lambert
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // Tall block    -- white lambert
    3, 3                           // Ceiling light -- emmissive
}};

static size_t roundUp(size_t x, size_t y) {
    return ( ( x + y - 1 ) / y ) * y;
}

void buildMeshAccel2(PathTracerHackyState* state, void* scn) {
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

    Scene* scene = reinterpret_cast<Scene*>(scn);

    std::vector<Vertex>          g_vertices;
    std::vector<IndexedTriangle> g_indices;

    const std::vector<Object*>* objects = scene->getObjects();
    for (Object* object : *objects) {
        Mesh* mesh = static_cast<Mesh*>(object);
        int32_t tri_count = mesh->getTriangleCount();
        uint32_t max_idx = 0;
        for (int32_t i = 0; i < tri_count; i++) {
            Eigen::Vector3i indices = mesh->getTriangleIndices(i);
            uint32_t a = indices(0), b = indices(1), c =indices(2);
            uint32_t max_cur = a > b ?
                                   a > c ?
                                       a : c : b;
            max_idx = max_cur > max_idx ? max_cur : max_idx;

            g_indices.push_back({a, b, c, 0});
        }
        for (uint32_t i = 0; i < max_idx; i++) {
            Eigen::Vector3f vertex = mesh->getVertex(i);
            float a = vertex(0), b = vertex(1), c = vertex(2);
            g_vertices.push_back({a, b, c, 0.f});
        }
    }

    // Copy vertex data to device
    const size_t vertices_size = sizeof(Vertex) * g_vertices.size();
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state->d_vertices), vertices_size));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(state->d_vertices),
        g_vertices.data(),
        vertices_size,
        cudaMemcpyHostToDevice
        ));

    const size_t indices_size = sizeof(IndexedTriangle) * g_indices.size();
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state->d_indices), indices_size));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(state->d_indices),
        g_indices.data(),
        indices_size,
        cudaMemcpyHostToDevice
        ));

    // Matrix data to device
    CUdeviceptr d_mat_indices = 0;
    const size_t mat_indices_size_in_bytes = g_mat_indices.size() * sizeof(uint32_t);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_mat_indices), mat_indices_size_in_bytes));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_mat_indices),
        g_mat_indices.data(),
        mat_indices_size_in_bytes,
        cudaMemcpyHostToDevice
        ));

    // Build triangle GAS
    const uint32_t triangle_input_flags[4] = {
        OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
        OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
        OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
        OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT
    };

    OptixBuildInput triangle_input = {};
    triangle_input.type                                      = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat                = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.vertexStrideInBytes         = sizeof(Vertex);
    triangle_input.triangleArray.numVertices                 = uint32_t(g_vertices.size());
    triangle_input.triangleArray.vertexBuffers               = &state->d_vertices;
    triangle_input.triangleArray.numIndexTriplets            = uint32_t(g_indices.size());
    triangle_input.triangleArray.indexBuffer                 = state->d_indices;
    triangle_input.triangleArray.indexFormat                 = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangle_input.triangleArray.indexStrideInBytes          = sizeof(IndexedTriangle);
    triangle_input.triangleArray.flags                       = triangle_input_flags;
    triangle_input.triangleArray.numSbtRecords               = 4;
    triangle_input.triangleArray.sbtIndexOffsetBuffer        = d_mat_indices;
    triangle_input.triangleArray.sbtIndexOffsetSizeInBytes   = sizeof(uint32_t);
    triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        state->context,
        &accel_options,
        &triangle_input,
        1,  // number of build inputs
        &gas_buffer_sizes
        ));

    CUdeviceptr d_temp_buffer_gas;
    CUDA_CHECK( cudaMalloc(
        reinterpret_cast<void**>(&d_temp_buffer_gas),
        gas_buffer_sizes.tempSizeInBytes
        ));

    // buffer for non-compacted output
    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t compactedSizeOffset = roundUp(gas_buffer_sizes.outputSizeInBytes, 8ULL);
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_buffer_temp_output_gas_and_compacted_size),
        compactedSizeOffset + 8
        ));

    OptixAccelEmitDesc emit_property = {};
    emit_property.type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emit_property.result             = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

    OPTIX_CHECK(optixAccelBuild(
        state->context,
        0,  // CUDA stream
        &accel_options,
        &triangle_input,
        1,  // number of build inputs
        d_temp_buffer_gas,
        gas_buffer_sizes.tempSizeInBytes,
        d_buffer_temp_output_gas_and_compacted_size,
        gas_buffer_sizes.outputSizeInBytes,
        &state->gas_handle,
        &emit_property, // emitted property list
        1               // num emitted properties
        ));

    // Free scratch space (TODO: revisit)
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer_gas)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_mat_indices)));

    size_t compacted_gas_size;
    CUDA_CHECK(cudaMemcpy(&compacted_gas_size, reinterpret_cast<void*>(emit_property.result), sizeof(size_t), cudaMemcpyDeviceToHost));
    if(compacted_gas_size < gas_buffer_sizes.outputSizeInBytes) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state->d_gas_output_buffer), compacted_gas_size));
        // use handle as input and output
        OPTIX_CHECK(optixAccelCompact(
            state->context,
            0,
            state->gas_handle,
            state->d_gas_output_buffer,
            compacted_gas_size,
            &state->gas_handle
            ));
        CUDA_CHECK(cudaFree((void*)d_buffer_temp_output_gas_and_compacted_size));
    } else {
        state->d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }
}
