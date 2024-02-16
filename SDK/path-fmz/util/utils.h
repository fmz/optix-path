#ifndef UTILS_H
#define UTILS_H

#include <random>
#include <qrgb.h>
#include <vector>
#include <Eigen/Dense>

static constexpr float eps = 1e-6;

static std::mt19937 global_rng(0);

static float getRandFloat(float min = 0.f, float max = 1.f, std::mt19937& rng = global_rng) {
    float range = max - min;
    return float(double(rng()) / rng.max()) * range + min;
}

static int32_t getRandInt(int32_t min = 0, int32_t max = INT_MAX, std::mt19937& rng = global_rng) {
    std::uniform_int_distribution<> distrib(min, max-1);
    return distrib(global_rng);
}

static int32_t sampleFromDist(const std::vector<float>& vec) {
    std::discrete_distribution<> dist(vec.begin(), vec.end());

    return dist(global_rng);
}


// Helper function to convert illumination to RGBA, applying some form of tone-mapping (e.g. clamping) in the process
static QRgb vec3ToQRGB(const Eigen::Vector3f& illum) {
    static const Eigen::Vector3f lo = Eigen::Vector3f::Constant(0.f);
    static const Eigen::Vector3f hi = Eigen::Vector3f::Constant(1.f);

    Eigen::Vector3f clamped = illum.cwiseMin(hi).cwiseMax(lo);
    Eigen::Vector3i intvec = (clamped * 255.f).cast<int32_t>();

    return qRgb(intvec[0], intvec[1], intvec[2]);
}

#endif // UTILS_H
