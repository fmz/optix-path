#include "pathtracer.h"

#include <iostream>
#include <vector>
#include <Eigen/Dense>

#include <util/CS123Common.h>

#include "util/utils.h"

using namespace Eigen;

PathTracer::PathTracer(int width, int height)
    : m_width(width), m_height(height)
{
}


std::pair<float, Vector3f> PathTracer::uniformSampleFromHemisphere(
    const Eigen::Vector3f& normal
) {
    // Sample a theta w.r.t the orthogonal (doesn't matter which orthogonal)
    Vector3f ortho;

    float x = normal(0), y = normal(1), z = normal(2);

    // Lifted from pbrt
    float sign = z >= 0.f ? 1.f : -1.f;
    float a = -1.f / (sign + z);
    float b = x * y * a;
    //ortho << (1 - (x*x/(1 + z))), -(x*y/(1 + z)), -x;
    ortho << b, sign + y*y * a, -y;
    ortho.normalize();

    float theta = getRandFloat(0, M_PI_2);
    // Sample phi w.r.t normal
    float phi   = getRandFloat(0, 2.f*M_PI);

    AngleAxisf theta_rot(theta, ortho);
    AngleAxisf phi_rot(phi, normal);

    Quaternionf q_theta(theta_rot);
    Quaternionf q_phi(phi_rot);

    Eigen::Vector3f out = q_phi * q_theta * normal;

    //static int64_t count = 0;
    // if (count++ < 100000) {
    //     // std::cout << "out = " << out << std::endl;
    //     // std::cout << "nrm = " << normal << std::endl;
    //     std::cout << "crs = " << ortho.dot(normal) << std::endl;
    // }
    if (ortho.dot(normal) > eps) {
        std::cerr << "ortho is not really ortho!" << std::endl;
    }

    return {1.f/(2*M_PI), out};
}

// Sample a light source.
std::pair<float, Eigen::Vector3f> PathTracer::sampleLight(
    const std::vector<Triangle*>& lights
) {
    int32_t idx = sampleFromDist(light_areas);

    // FIXME: acutally sample from the area
    const Triangle* tri = lights[idx];
    Matrix3f trimat;
    trimat << tri->_v1, tri->_v2, tri->_v3;

    float a = getRandFloat();
    float b = getRandFloat();
    float c = getRandFloat();

    Vector3f bc_coords{a, b, c};

    Vector3f ret = trimat * bc_coords;

    return {1.f / tae, ret};
}

void PathTracer::traceScene(QRgb *imageData, const Scene& scene) {
    // Before we do the loop, calculate any "meta" params
    _pix_subdivs = int32_t(ceil(sqrt(settings.samplesPerPixel)));
    _pix_subdiv_sz_scrn = 2.f / (_pix_subdivs*m_width);

    // The total area of our light sources
    const std::vector<Triangle*>& tris = scene.getEmissives();
    tae = 0.f;
    light_areas.reserve(tris.size());
    for (size_t i = 0; i < tris.size(); i++) {
        float area = tris[i]->getArea();
        light_areas[i] = area;
        tae += area;
    }

    std::vector<Vector3f> intensityValues(m_width * m_height);
    Matrix4f invViewMat = (scene.getCamera().getScaleMatrix() * scene.getCamera().getViewMatrix()).inverse();
    for(int y = 0; y < m_height; ++y) {
        //#pragma omp parallel for
        for(int x = 0; x < m_width; ++x) {
            int offset = x + (y * m_width);
            intensityValues[offset] = tracePixel(x, y, scene, invViewMat);
        }
    }

    toneMap(imageData, intensityValues);
}

Vector3f PathTracer::tracePixel(int x, int y, const Scene& scene, const Matrix4f &invViewMatrix)
{
    // Subdivide each pixel into ceil(sqrt(samplesPerPixel)) cells per dimension
    // (see _dim_subdiv calculation in traceScene)

    int32_t total_subdivs = _pix_subdivs * _pix_subdivs;
    Vector3f accum = Vector3f::Zero();
    for (float sub_y = 0.5f; sub_y <= _pix_subdivs; sub_y += 1.f) {
        float y_sub_offset = sub_y * _pix_subdiv_sz_scrn;
        for (float sub_x = 0.5f; sub_x <= _pix_subdivs; sub_x += 1.f) {
            float x_sub_offset = sub_x * _pix_subdiv_sz_scrn;

            Vector3f p(0, 0, 0);
            Vector3f d((2.f * x / m_width + x_sub_offset) - 1, 1 - (2.f * y / m_height + y_sub_offset), -1);
            d.normalize();

            Ray r(p, d);
            r = r.transform(invViewMatrix);
            accum = accum.array() + traceRay(r, scene).array();
        }
    }

    return accum / total_subdivs;
}


Vector3f PathTracer::traceRay(const Ray& r, const Scene& scene, int32_t lvl, bool accum_light) {
    IntersectionInfo i;
    Ray ray(r);
    Vector3f color(0.f, 0.f, 0.f);

    if (scene.getIntersection(ray, &i)) {
        // ** Example code for accessing materials provided by a .mtl file **
        const Triangle *t = static_cast<const Triangle *>(i.data);//Get the triangle in the mesh that was intersected
        const tinyobj::material_t& mat = t->getMaterial();//Get the material of the triangle from the mesh
        //const float illum = mat.illum;
        //const float shine = mat.shininess;
        // const Vector3f s(mat.specular[0] * shine, mat.specular[1] * shine, mat.specular[2] * shine);
        const Vector3f d(mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]);//Diffuse color as array of floats
        //d = d.array() + s.array();
        Vector3f normal = t->getNormal(i.hit);

        if (accum_light && (mat.emission[0] > 0.f || mat.emission[1] > 0.f || mat.emission[2] > 0.f))  {
            color = Vector3f(mat.emission).array();
        }

        if (accum_light && lvl > 0) {
            return color;
        }

        // Direct lighting ray
        std::pair<float, Vector3f> light_dir = sampleLight(scene.getEmissives());
        light_dir.second -= i.hit;
        const Ray light_ray(i.hit + light_dir.second * eps, light_dir.second);
        // if (lvl == 0) {
        //     std::cout << "new_ray = (" << new_ray.o.array() << ", " << new_ray.d.array() << std::endl;
        // }
        Vector3f new_color =
            d.array() *
            traceRay(light_ray, scene, lvl+1, true).array() *
            light_ray.d.dot(normal) /
            (light_dir.first);

        color += new_color;

        // Indirect ray
        float cont = getRandFloat();
        const float prob_rr = settings.pathContinuationProb;
        if (cont > prob_rr) {
            return color;
        }

        std::pair<float, Vector3f> new_dir = uniformSampleFromHemisphere(normal);
        const Ray new_ray(i.hit + new_dir.second * eps, new_dir.second);
        // if (lvl == 0) {
        //     std::cout << "new_ray = (" << new_ray.o.array() << ", " << new_ray.d.array() << std::endl;
        // }
        new_color =
            d.array() *
            traceRay(new_ray, scene, lvl+1, false).array() *
            new_ray.d.dot(normal) /
            (new_dir.first * prob_rr * M_PI /*brdf normalization*/);

        color += new_color;
    }
    return color;
}

static float getLuminance(const Eigen::Vector3f& vec) {
    return vec(0) * 0.2126f + vec(1) * 0.7152f + vec(2) * 0.0722f;
}

void PathTracer::toneMap(QRgb *imageData, std::vector<Vector3f> &intensityValues) {
    //outputPFM("pfm.pfm", 512, 512, intensityValues);
    // Find the most intense color
    float max_lum = 0;
    for (int y = 0; y < m_height; ++y) {
        for(int x = 0; x < m_width; ++x) {
            int offset = x + (y * m_width);
            const Vector3f& intn = intensityValues[offset];
            float lum = getLuminance(intn);
            if (lum > max_lum) {
                max_lum = lum;
            }
        }
    }

    // Needs to be squared
    max_lum = max_lum * max_lum;
    static const Vector3f gamma = Vector3f::Ones() / 2.2f;

    for(int y = 0; y < m_height; ++y) {
        for(int x = 0; x < m_width; ++x) {
            int offset = x + (y * m_width);
            //imageData[offset] = intensityValues[offset].norm() > 0 ? qRgb(255, 255, 255) : qRgb(40, 40, 40);
            const Vector3f& intn = intensityValues[offset];

            // Calculate extended reinhard

            // Find the output luminance
            float lum_in = getLuminance(intn);
            float lum_out = lum_in * (1.f + (lum_in / max_lum));
            lum_out = lum_out / (1.0f + lum_in);
            // Update the luminance
            Vector3f reinhard = intn * (lum_out / lum_in);
            reinhard = reinhard.array().pow(gamma.array());

            // non-extended reinhard
            //Vector3f reinhard = intn.array()  / (1.f + intn.array());
            imageData[offset] = vec3ToQRGB(reinhard);

        }
    }
}
