#ifndef PATHTRACERINTERFACE_H
#define PATHTRACERINTERFACE_H

#include <QImage>
#include <QtCore>

//#include "scene/scene.h"
#include "path-tracer.h"

int32_t runTracer(const TracerSettings& settings,
    void* scene,
    QRgb* data_out,
    const std::string out_filename);

#endif // PATHTRACERINTERFACE_H
