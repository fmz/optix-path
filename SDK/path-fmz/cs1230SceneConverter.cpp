#include <cstdint>
#include <map>

// CUDA
#include <cuda_runtime.h>

#include "scene/scene.h"
#include "cs1230SceneConverter.h"

static inline float3 eigen_to_vec_f(const Eigen::Vector3f& v) {
    return {v(0), v(1), v(2)};
}

template <typename T>
static inline T max(const T& a, const T& b, const T& c) {
    return a > b ? a > c ? a : c : b > c ? b : c;
}

SceneConverter::SceneConverter(void* scn) {
    // HACK!
    Scene* scene = reinterpret_cast<Scene*>(scn);
    const std::vector<Object*>* objects = scene->getObjects();
    if (objects->size() > 1) {
        throw std::runtime_error("can't handle multiple objects in one mesh for now!");
    }
    for (Object* object : *objects) {
        Mesh* mesh = static_cast<Mesh*>(object);
        int32_t tri_count = mesh->getTriangleCount();
        uint32_t max_idx = 0;

        // Indices
        for (int32_t i = 0; i < tri_count; i++) {
            Eigen::Vector3i eig_idx = mesh->getTriangleIndices(i);
            uint32_t a = eig_idx(0), b = eig_idx(1), c = eig_idx(2);
            uint32_t max_cur = max(a, b, c);
            max_idx = max_cur > max_idx ? max_cur : max_idx;

            indices.push_back({a, b, c, 0});
        }

        // Vertices and normals
        for (uint32_t i = 0; i <= max_idx; i++) {
            Eigen::Vector3f vertex = mesh->getVertex(i);
            Eigen::Vector3f normal = mesh->getNormal(i);
            float a = vertex(0), b = vertex(1), c = vertex(2);
            vertices.push_back({a, b, c, 0.f});
            normals.push_back(eigen_to_vec_f(normal));
        }

        // Also consider collapsing the loop below with the ones above

        // Colors and emissives
        // TODO: use unordered_map to speed up lookup

        struct float6 {
            float3 a, b;
        };

        auto comp = [](const float6& a, const float6& b) {
            return a.a.x < b.a.x ||
                   a.a.y < b.a.y ||
                   a.a.z < b.a.z ||
                   a.b.x < b.b.x ||
                   a.b.y < b.b.y ||
                   a.b.z < b.b.z;
        };
        std::map<float6, uint32_t, decltype(comp)> emiss_to_idx;

        bool light_found = false;

        for (int32_t i = 0; i < tri_count; i++) {
            const auto& mat = mesh->getMaterial(i);
            float3 emiss = {
                mat.emission[0],
                mat.emission[1],
                mat.emission[2]
            };

            float3 color = {
                mat.diffuse[0],
                mat.diffuse[1],
                mat.diffuse[2]
            };

            float6 pack{emiss, color};
            if (emiss_to_idx.count(pack) == 0) {
                emission.push_back(emiss);
                diffuse.push_back(color);
                mat_indices.push_back(emission.size() - 1);
                emiss_to_idx[pack] = emission.size() - 1;
            } else {
                mat_indices.push_back(emiss_to_idx.at(pack));
            }

            // Handle the one light when you find it
            // TODO: keep track of a list of emissive triangles and their areas instead
            if (!light_found && (emiss.x > 0 || emiss.y > 0 || emiss.z > 0)) {
                Eigen::Vector3i tri_vs = mesh->getTriangleIndices(i);
                Eigen::Vector3f v0 = mesh->getVertex(tri_vs(0));
                Eigen::Vector3f v1 = mesh->getVertex(tri_vs(1));
                Eigen::Vector3f v2 = mesh->getVertex(tri_vs(2));

                v1 = v0 - v1;
                v2 = v0 - v2;

                Eigen::Vector3f normal = v1.cross(v2).normalized();


                light.emission = emiss;
                light.corner = eigen_to_vec_f(v0);
                light.v1     = eigen_to_vec_f(v1);
                light.v2     = eigen_to_vec_f(v2);
                light.normal = eigen_to_vec_f(normal);

                light_found = true;
            }
        }
    }

    mat_count = diffuse.size();
    tri_count = indices.size();


    // Camera
    const BasicCamera& cam = scene->getCamera();

    const Eigen::Vector3f& pos = cam.m_position;
    eye = eigen_to_vec_f(pos);

    const Eigen::Vector3f look_at = cam.m_direction + cam.m_position;
    lookat = eigen_to_vec_f(look_at);

    const Eigen::Vector3f foq = cam.m_up;
    up = eigen_to_vec_f(foq);
}
