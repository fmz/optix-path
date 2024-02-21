// Std library
#include <cstdint>
#include <string>
#include <iostream>
#include <iomanip>

// CUDA
#include <glad/glad.h>  // Needs to be included before gl_interop
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

// Optix
#include <optix.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

// GLFW
#include <GLFW/glfw3.h>

// Internal includes
#include "path-tracer-interface.h"
#include "sutil/sutil.h"
#include "sutil/Exception.h"
#include "sutil/GLDisplay.h"
#include "sutil/Trackball.h"
#include "sutil/CUDAOutputBuffer.h"
#include "sutil/Camera.h"

#include "cs1230SceneConverter.h"

//#define DEBUG 1

bool resize_dirty = false;
bool minimized    = false;

// Camera
bool camera_changed = true;
sutil::Camera    camera;
sutil::Trackball trackball;

// Mouse
int32_t mouse_button = -1;

template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData>   RayGenSbtRecord;
typedef SbtRecord<MissData>     MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

struct PathTracerState {
    OptixDeviceContext context = 0;

    // Accel state
    OptixTraversableHandle gas_handle          = 0;
    CUdeviceptr            d_gas_output_buffer = 0;
    CUdeviceptr            d_vertices          = 0;
    CUdeviceptr            d_indices           = 0;
    CUdeviceptr            d_normals           = 0;


    OptixModule                 optix_module = 0;
    OptixPipelineCompileOptions pco          = {};
    OptixPipeline               pipeline     = 0;

    OptixProgramGroup raygen_prog_group   = 0;
    OptixProgramGroup radiance_miss_group = 0;
    OptixProgramGroup radiance_hit_group  = 0;

    CUstream stream = 0;

    Params  params;
    Params* d_params;

    OptixShaderBindingTable sbt = {};
};

// GLFW callbacks
static void mouseButtonCallback(GLFWwindow* window, int32_t button, int32_t action, int32_t mods) {
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    if (action == GLFW_PRESS) {
        mouse_button = button;
        trackball.startTracking(int32_t(xpos), int32_t(ypos));
    } else {
        mouse_button = -1;
    }
}

static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    Params* params = static_cast<Params*>(glfwGetWindowUserPointer(window));

    if (mouse_button == GLFW_MOUSE_BUTTON_LEFT) {
        trackball.setViewMode(sutil::Trackball::LookAtFixed);
        trackball.updateTracking(int32_t(xpos), int32_t(ypos), params->width, params->height);
        camera_changed = true;
    } else if (mouse_button == GLFW_MOUSE_BUTTON_RIGHT) {
        trackball.setViewMode(sutil::Trackball::EyeFixed);
        trackball.updateTracking(int32_t(xpos), int32_t(ypos), params->width, params->height);
        camera_changed = true;
    } // else do nothing
}

static void windowSizeCallback(GLFWwindow* window, int32_t res_x, int32_t res_y) {
    if (minimized) {
        return;
    }

    // Output dimensions must be at least 1 in x and y
    sutil::ensureMinimumSize(res_x, res_y);

    Params* params = static_cast<Params*>(glfwGetWindowUserPointer(window));
    params->width  = res_x;
    params->height = res_y;
    camera_changed = true;
    resize_dirty   = true;
}

static void windowIconifyCallback(GLFWwindow* window, int32_t iconified) {
    minimized = (iconified > 0);
}

static void keyCallback(GLFWwindow* window, int32_t key, int32_t scancode, int32_t action, int32_t mods) {
    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE) {
            glfwSetWindowShouldClose(window, true);
        }
        // TODO: add other keys
    }
}

static void scrollCallback(GLFWwindow* window, double xscroll, double yscroll) {
    if (trackball.wheelEvent(int32_t(yscroll))) {
        camera_changed = true;
    }
}


// Helpers
void handleCameraUpdate(Params& params) {
    if(!camera_changed) {
        return;
    }
    camera_changed = false;

    camera.setAspectRatio(float(params.width) / float(params.height));
    params.cam_eye = camera.eye();
    camera.UVWFrame(params.cam_u, params.cam_v, params.cam_w);
}

void handleResize(sutil::CUDAOutputBuffer<uchar4>& output_buffer, Params& params) {
    if(!resize_dirty) {
        return;
    }
    resize_dirty = false;

    output_buffer.resize(params.width, params.height);

    // Reallocate accumulation buffer
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(params.accum_buffer)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&params.accum_buffer),
        params.width * params.height * sizeof(float4)
    ));
}

void updateState(sutil::CUDAOutputBuffer<uchar4>& output_buffer, Params& params) {
    // Update params on device
    if(camera_changed || resize_dirty) {
        params.subframe_index = 0;
    }

    handleCameraUpdate(params);
    handleResize(output_buffer, params);
}

void launchSubframe(sutil::CUDAOutputBuffer<uchar4>& output_buffer, PathTracerState& state) {
    // Launch
    uchar4* result_buffer_data = output_buffer.map();
    state.params.frame_buffer  = result_buffer_data;
    CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(state.d_params),
        &state.params, sizeof(Params),
        cudaMemcpyHostToDevice, state.stream
    ));

    OPTIX_CHECK(optixLaunch(
        state.pipeline,
        state.stream,
        reinterpret_cast<CUdeviceptr>(state.d_params),
        sizeof(Params),
        &state.sbt,
        state.params.width,   // launch width
        state.params.height,  // launch height
        1                     // launch depth
    ));

    output_buffer.unmap();
    CUDA_SYNC_CHECK();
}

void displaySubframe(sutil::CUDAOutputBuffer<uchar4>& output_buffer, sutil::GLDisplay& gl_display, GLFWwindow* window) {
    // Display
    int framebuf_res_x = 0;  // The display's resolution (could be HDPI res)
    int framebuf_res_y = 0;  //
    glfwGetFramebufferSize(window, &framebuf_res_x, &framebuf_res_y);
    gl_display.display(
        output_buffer.width(),
        output_buffer.height(),
        framebuf_res_x,
        framebuf_res_y,
        output_buffer.getPBO()
    );
}

void initLaunchParams(PathTracerState& state, const TracerSettings& settings, SceneConverter& sc) {
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&state.params.accum_buffer),
        state.params.width * state.params.height * sizeof(float4)
    ));

    state.params.frame_buffer = nullptr;    // Will be mapped later

    state.params.samples_per_launch = settings.samplesPerPixel;
    state.params.continuation_prob  = settings.pathContinuationProb;
    state.params.direct_light_only  = settings.directLightingOnly;
    state.params.subframe_index     = 0u;

    state.params.light = sc.light;

    state.params.handle = state.gas_handle;

    CUDA_CHECK(cudaStreamCreate(&state.stream));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_params), sizeof(Params)));
}

void initCameraState(SceneConverter& sc) {

    camera.setEye(sc.eye);
    camera.setLookat(sc.lookat);
    camera.setUp(sc.up);
    camera.setFovY(45.0f);
    camera_changed = true;

    trackball.setCamera( &camera );
    trackball.setMoveSpeed( 10.0f );
    trackball.setReferenceFrame(
        make_float3( 1.0f, 0.0f, 0.0f ),
        make_float3( 0.0f, 0.0f, 1.0f ),
        make_float3( 0.0f, 1.0f, 0.0f )
    );
    trackball.setGimbalLock( true );
}

static void ctx_log_cb(uint32_t level, const char* tag, const char* message, void*) {
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << std::endl;
}

void createContext(PathTracerState& state) {
    // Kick CUDA to initialize it
    CUDA_CHECK(cudaFree(0));

    OptixDeviceContext context;
    CUcontext cuCtx = 0;
    OPTIX_CHECK(optixInit());

    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &ctx_log_cb;
    options.logCallbackLevel          = 4;
#ifdef DEBUG
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));

    state.context = context;
}

void buildMeshAccel(PathTracerState& state, SceneConverter& sc) {
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

    // Copy vertex data to device
    const size_t vertices_size = sizeof(Vertex) * sc.vertices.size();
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_vertices), vertices_size));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(state.d_vertices),
        sc.vertices.data(),
        vertices_size,
        cudaMemcpyHostToDevice
    ));

    // Copy index data to device
    const size_t indices_size = sizeof(IndexedTriangle) * sc.indices.size();
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_indices), indices_size));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(state.d_indices),
        sc.indices.data(),
        indices_size,
        cudaMemcpyHostToDevice
    ));

    // Normals don't really need to be copied here, but why not...
    const size_t normals_size = sizeof(float3) * sc.normals.size();
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_normals), normals_size));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(state.d_normals),
        sc.normals.data(),
        normals_size,
        cudaMemcpyHostToDevice
    ));

    // Matrix data to device
    CUdeviceptr d_mat_indices = 0;
    const size_t mat_indices_size_in_bytes = sc.mat_indices.size() * sizeof(uint32_t);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_mat_indices), mat_indices_size_in_bytes));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_mat_indices),
        sc.mat_indices.data(),
        mat_indices_size_in_bytes,
        cudaMemcpyHostToDevice
    ));

    // Build triangle GAS
    uint32_t triangle_input_flags[sc.mat_count];
    for (size_t i = 0; i < sc.mat_count; i++) {
        triangle_input_flags[i] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
    };

    OptixBuildInput triangle_input = {};
    triangle_input.type                                      = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat                = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.vertexStrideInBytes         = sizeof(Vertex);
    triangle_input.triangleArray.numVertices                 = uint32_t(sc.vertices.size());
    triangle_input.triangleArray.vertexBuffers               = &state.d_vertices;
    triangle_input.triangleArray.numIndexTriplets            = uint32_t(sc.indices.size());
    triangle_input.triangleArray.indexBuffer                 = state.d_indices;
    triangle_input.triangleArray.indexFormat                 = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangle_input.triangleArray.indexStrideInBytes          = sizeof(IndexedTriangle);
    triangle_input.triangleArray.flags                       = triangle_input_flags;
    triangle_input.triangleArray.numSbtRecords               = sc.mat_count;
    triangle_input.triangleArray.sbtIndexOffsetBuffer        = d_mat_indices;
    triangle_input.triangleArray.sbtIndexOffsetSizeInBytes   = sizeof(uint32_t);
    triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        state.context,
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
    size_t compactedSizeOffset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ULL);
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_buffer_temp_output_gas_and_compacted_size),
        compactedSizeOffset + 8
    ));

    OptixAccelEmitDesc emit_property = {};
    emit_property.type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emit_property.result             = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

    OPTIX_CHECK(optixAccelBuild(
        state.context,
        0,  // CUDA stream
        &accel_options,
        &triangle_input,
        1,  // number of build inputs
        d_temp_buffer_gas,
        gas_buffer_sizes.tempSizeInBytes,
        d_buffer_temp_output_gas_and_compacted_size,
        gas_buffer_sizes.outputSizeInBytes,
        &state.gas_handle,
        &emit_property, // emitted property list
        1               // num emitted properties
    ));

    // Free scratch space (TODO: revisit)
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer_gas)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_mat_indices)));

    size_t compacted_gas_size;
    CUDA_CHECK(cudaMemcpy(&compacted_gas_size, reinterpret_cast<void*>(emit_property.result), sizeof(size_t), cudaMemcpyDeviceToHost));
    if(compacted_gas_size < gas_buffer_sizes.outputSizeInBytes) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_gas_output_buffer), compacted_gas_size));
        // use handle as input and output
        OPTIX_CHECK(optixAccelCompact(
            state.context,
            0,
            state.gas_handle,
            state.d_gas_output_buffer,
            compacted_gas_size,
            &state.gas_handle
        ));
        CUDA_CHECK(cudaFree((void*)d_buffer_temp_output_gas_and_compacted_size));
    } else {
        state.d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }
}

void createModule(PathTracerState& state) {
    OptixPayloadType payloadType = {};
    // radiance prd
    payloadType.numPayloadValues = sizeof(radiancePayloadSemantics) / sizeof(radiancePayloadSemantics[0]);
    payloadType.payloadSemantics = radiancePayloadSemantics;

    OptixModuleCompileOptions mco = {};
#if !defined( NDEBUG )
    mco.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    mco.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif
    mco.numPayloadTypes = 1;
    mco.payloadTypes    = &payloadType;

    state.pco.usesMotionBlur                   = false;
    state.pco.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    state.pco.numPayloadValues                 = 0;
    state.pco.numAttributeValues               = 2;
    state.pco.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
    state.pco.pipelineLaunchParamsVariableName = "params";

    size_t      inputSize = 0;
    const char* input     = sutil::getInputData(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "path-tracer.cu", inputSize);

    OPTIX_CHECK_LOG(optixModuleCreate(
        state.context,
        &mco,
        &state.pco,
        input,
        inputSize,
        LOG, &LOG_SIZE,
        &state.optix_module
    ));
}

void createProgramGroups(PathTracerState& state) {
    // Create program groups
    OptixProgramGroupOptions pgo = {};
    {
        OptixProgramGroupDesc raygen_pgd = {};
        raygen_pgd.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_pgd.raygen.module            = state.optix_module;
        raygen_pgd.raygen.entryFunctionName = "__raygen__path_tracer";
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            state.context,
            &raygen_pgd,
            1,
            &pgo,
            LOG,
            &LOG_SIZE,
            &state.raygen_prog_group
        ));
    }

    {
        OptixProgramGroupDesc miss_pgd  = {};
        miss_pgd.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_pgd.miss.module            = state.optix_module;
        miss_pgd.miss.entryFunctionName = "__miss__radiance";
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            state.context,
            &miss_pgd,
            1,
            &pgo,
            LOG,
            &LOG_SIZE,
            &state.radiance_miss_group
        ));
    }
    {
        OptixProgramGroupDesc hitgroup_pgd    = {};
        hitgroup_pgd.kind                     = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitgroup_pgd.hitgroup.moduleCH        = state.optix_module;     // can assign different modules for closest-hit, any-hit, and intersection
        hitgroup_pgd.raygen.entryFunctionName = "__closesthit__radiance";
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            state.context,
            &hitgroup_pgd,
            1,
            &pgo,
            LOG,
            &LOG_SIZE,
            &state.radiance_hit_group
        ));
    }
}

void createPipeline(PathTracerState& state) {
    // Linking stage
    OptixProgramGroup program_groups[] = {
        state.raygen_prog_group,
        state.radiance_miss_group,
        state.radiance_hit_group
    };

    OptixPipelineLinkOptions plo = {};
    plo.maxTraceDepth            = 2;       // Not sure why this is different from max_trace_depth below
    OPTIX_CHECK_LOG(optixPipelineCreate(
        state.context,
        &state.pco,
        &plo,
        program_groups,
        sizeof(program_groups) / sizeof(program_groups[0]),
        LOG,
        &LOG_SIZE,
        &state.pipeline
    ));

    // We need to specify the max traversal depth.  Calculate the stack sizes, so we can specify all
    // parameters to optixPipelineSetStackSize.
    OptixStackSizes stack_sizes = {};
    for (const auto& grp : program_groups) {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(grp, &stack_sizes, state.pipeline));
    }

    uint32_t max_trace_depth = 2;
    uint32_t max_cc_depth    = 0;
    uint32_t max_dc_depth    = 0;
    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;

    OPTIX_CHECK(optixUtilComputeStackSizes(
        &stack_sizes,
        max_trace_depth,
        max_cc_depth,
        max_dc_depth,
        &direct_callable_stack_size_from_traversal,
        &direct_callable_stack_size_from_state,
        &continuation_stack_size
    ));

    OPTIX_CHECK(optixPipelineSetStackSize(
        state.pipeline,
        direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state,
        continuation_stack_size,
        1  // maxTraversableDepth
    ));
}

void createSBT(PathTracerState& state, SceneConverter& sc) {
    CUdeviceptr d_raygen_record;
    const size_t raygen_record_size = sizeof(RayGenSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygen_record), raygen_record_size));
    RayGenSbtRecord rg_sbt = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(state.raygen_prog_group, &rg_sbt));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_raygen_record),
        &rg_sbt,
        raygen_record_size,
        cudaMemcpyHostToDevice
    ));

    CUdeviceptr d_miss_record;
    const size_t miss_record_size = sizeof(MissSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_miss_record), miss_record_size * RAY_TYPE_COUNT));
    MissSbtRecord miss_sbt[RAY_TYPE_COUNT];
    miss_sbt[0].data.bg_clr = {0.f, 0.f, 0.f, 0.f};
    OPTIX_CHECK(optixSbtRecordPackHeader(state.radiance_miss_group, &miss_sbt[0]));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_miss_record),
        miss_sbt,
        miss_record_size * RAY_TYPE_COUNT,
        cudaMemcpyHostToDevice
    ));


    CUdeviceptr d_hitgroup_records;
    const size_t hitgroup_record_size = sizeof(HitGroupSbtRecord);
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_hitgroup_records),
        hitgroup_record_size * RAY_TYPE_COUNT * sc.mat_count
    ));

    HitGroupSbtRecord hg_sbt_records[RAY_TYPE_COUNT * sc.mat_count];
    for (size_t i = 0; i < sc.mat_count; i++) {
        const size_t sbt_idx = i * RAY_TYPE_COUNT;
        OPTIX_CHECK(optixSbtRecordPackHeader(state.radiance_hit_group, &hg_sbt_records[sbt_idx]));
        hg_sbt_records[sbt_idx].data.mat_info = sc.mat_info[i];
        hg_sbt_records[sbt_idx].data.vertices = reinterpret_cast<float4*>(state.d_vertices);
        hg_sbt_records[sbt_idx].data.normals  = reinterpret_cast<float3*>(state.d_normals);
    }

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_hitgroup_records),
        &hg_sbt_records,
        hitgroup_record_size * RAY_TYPE_COUNT * sc.mat_count,
        cudaMemcpyHostToDevice
    ));

    state.sbt.raygenRecord = d_raygen_record;

    state.sbt.missRecordBase          = d_miss_record;
    state.sbt.missRecordCount         = 1;
    state.sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);

    state.sbt.hitgroupRecordBase          = d_hitgroup_records;
    state.sbt.hitgroupRecordCount         = RAY_TYPE_COUNT * sc.mat_count;
    state.sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
}

void cleanupState(PathTracerState& state) {
    OPTIX_CHECK(optixPipelineDestroy(state.pipeline));
    OPTIX_CHECK(optixProgramGroupDestroy(state.raygen_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.radiance_miss_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.radiance_hit_group));
    OPTIX_CHECK(optixModuleDestroy(state.optix_module));
    OPTIX_CHECK(optixDeviceContextDestroy(state.context));

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.hitgroupRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.d_vertices)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.d_gas_output_buffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.params.accum_buffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.d_params)));
}

int32_t runTracer(
    const TracerSettings& settings,
    void* scene,
    QRgb* data_out,
    const std::string out_filename
 ) {
    PathTracerState state;
    state.params.width  = 512;
    state.params.height = 512;

    sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::GL_INTEROP;
    // output_buffer_type = sutil::CUDAOutputBufferType::CUDA_DEVICE;
    try {
        SceneConverter sc(scene);

        // Cam
        initCameraState(sc);

        // Optix state
        createContext(state);
        buildMeshAccel(state, sc);
        createModule(state);
        createProgramGroups(state);
        createPipeline(state);
        createSBT(state, sc);
        initLaunchParams(state, settings, sc);

        // Launch
        if (data_out == nullptr) {
            GLFWwindow* window = sutil::initUI( "RTX - ON", state.params.width, state.params.height );
            glfwSetMouseButtonCallback( window, mouseButtonCallback );
            glfwSetCursorPosCallback( window, cursorPosCallback );
            glfwSetWindowSizeCallback( window, windowSizeCallback );
            glfwSetWindowIconifyCallback( window, windowIconifyCallback );
            glfwSetKeyCallback( window, keyCallback );
            glfwSetScrollCallback( window, scrollCallback );
            glfwSetWindowUserPointer( window, &state.params );

            // Render loop
            {
                sutil::CUDAOutputBuffer<uchar4> output_buffer(
                    output_buffer_type,
                    state.params.width,
                    state.params.height
                );

                output_buffer.setStream(state.stream);
                sutil::GLDisplay gl_display;

                std::chrono::duration<double> state_update_time( 0.0 );
                std::chrono::duration<double> render_time( 0.0 );
                std::chrono::duration<double> display_time( 0.0 );

                do {
                    auto t0 = std::chrono::steady_clock::now();
                    glfwPollEvents();

                    updateState( output_buffer, state.params );
                    auto t1 = std::chrono::steady_clock::now();
                    state_update_time += t1 - t0;
                    t0 = t1;

                    launchSubframe( output_buffer, state );
                    t1 = std::chrono::steady_clock::now();
                    render_time += t1 - t0;
                    t0 = t1;

                    displaySubframe( output_buffer, gl_display, window );
                    t1 = std::chrono::steady_clock::now();
                    display_time += t1 - t0;

                    sutil::displayStats( state_update_time, render_time, display_time );

                    glfwSwapBuffers( window );

                    ++state.params.subframe_index;
                } while( !glfwWindowShouldClose( window ) );
                CUDA_SYNC_CHECK();
            }

            sutil::cleanupUI( window );
        } else {
            if( output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP ) {
                sutil::initGLFW();  // For GL context
                sutil::initGL();
            }

            {
                // this scope is for output_buffer, to ensure the destructor is called bfore glfwTerminate()

                sutil::CUDAOutputBuffer<uchar4> output_buffer(
                    output_buffer_type,
                    state.params.width,
                    state.params.height
                    );

                handleCameraUpdate( state.params );
                handleResize( output_buffer, state.params );
                launchSubframe( output_buffer, state );

                sutil::ImageBuffer buffer;
                buffer.data         = output_buffer.getHostPointer();
                buffer.width        = output_buffer.width();
                buffer.height       = output_buffer.height();
                buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;

                //data_out = reinterpret_cast<QRgb*>(output_buffer.getHostPointer());
                sutil::saveImage(out_filename.c_str(), buffer, false );
                std::cout << "Wrote rendered image to " << out_filename << std::endl;
            }

            if( output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP ) {
                glfwTerminate();
            }
        }

        cleanupState( state );
    } catch (std::exception& e) {
        std::cerr << "Caught exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
