#ifndef PATHTRACER_H
#define PATHTRACER_H

#include <QImage>
#include <vector>

#include "scene/scene.h"

struct Settings {
    int samplesPerPixel;
    bool directLightingOnly;      // if true, ignore indirect lighting
    int numDirectLightingSamples; // number of shadow rays to trace from each intersection point
    float pathContinuationProb;   // probability of spawning a new secondary ray == (1-pathTerminationProb)
};

class PathTracer
{
public:
    PathTracer(int width, int height);

    void traceScene(QRgb *imageData, const Scene &scene);
    Settings settings;

private:
    // Meta-settings: stuff that is computed from the settings
    int32_t _pix_subdivs;
    float _pix_subdiv_sz_scrn; // increment within each pixel in screen coords

    int32_t m_width, m_height;

    // (total_area_of_emissives) need to keep track of this for sampling purposes
    float tae;
    std::vector<float> light_areas;

    void toneMap(QRgb *imageData, std::vector<Eigen::Vector3f> &intensityValues);

    std::pair<float, Eigen::Vector3f> uniformSampleFromHemisphere(const Eigen::Vector3f& normal);
    std::pair<float, Eigen::Vector3f> sampleLight(
        const std::vector<Triangle*>& lights
    );
    Eigen::Vector3f tracePixel(int x, int y, const Scene &scene, const Eigen::Matrix4f &invViewMatrix);
    Eigen::Vector3f traceRay(const Ray& r, const Scene &scene, int32_t lvl = 0, bool accum_light=true);
};

#endif // PATHTRACER_H
