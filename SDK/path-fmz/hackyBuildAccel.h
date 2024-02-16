#ifndef HACKYBUILDACCEL_H
#define HACKYBUILDACCEL_H
#include <cstdint>
struct Vertex {
    float x, y, z, pad;
};

struct IndexedTriangle {
    uint32_t v1, v2, v3, pad;
};

struct Instance {
    float transform[12];
};

// This is needed because of lerp issues
// I can't stress this enough, this is a hack
struct PathTracerHackyState;
void buildMeshAccel2(PathTracerHackyState* state, void* scene);

#endif // HACKYBUILDACCEL_H
