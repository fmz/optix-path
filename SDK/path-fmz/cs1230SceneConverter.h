#ifndef CS1230SCENECONVERTER_H
#define CS1230SCENECONVERTER_H

// This whole setup is needed because of lerp issues

// Std library
#include <cstdint>
#include <vector>

// CUDA
#include <cuda_runtime.h>

#include "path-tracer.h"

struct Vertex {
    float x, y, z, pad;
};

struct IndexedTriangle {
    uint32_t v1, v2, v3, pad;
};

struct Instance {
    float transform[12];
};


// struct PathTracerHackyState;
// void buildMeshAccel2(PathTracerHackyState* state, void* scene);

struct SceneConverter {
    // This is a hack: needs to be Scene&, but lerp be lerpin...
    // Use namespaces when writing library code, people -_- (TODO: fix)
    SceneConverter(void* scene);

    // Mesh stuff
    std::vector<Vertex>           vertices;
    std::vector<IndexedTriangle>  indices;
    std::vector<float3>           normals;
    std::vector<uint32_t>         mat_indices;
    std::vector<float3>           emission;
    std::vector<float3>           diffuse;

    size_t mat_count;
    size_t tri_count;

    // Lights (hack for now: single parallelogram light)
    ParallelogramLight light;

    // Camera stuff
    float3 eye;
    float3 lookat;
    float3 up;
};

#endif // CS1230SCENECONVERTER_H
