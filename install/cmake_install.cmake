# Install script for directory: /home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/SDK

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/optixBoundValues/cmake_install.cmake")
  include("/home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/optixCallablePrograms/cmake_install.cmake")
  include("/home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/optixCompileWithTasks/cmake_install.cmake")
  include("/home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/optixConsole/cmake_install.cmake")
  include("/home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/optixCurves/cmake_install.cmake")
  include("/home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/optixCustomPrimitive/cmake_install.cmake")
  include("/home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/optixCutouts/cmake_install.cmake")
  include("/home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/optixDenoiser/cmake_install.cmake")
  include("/home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/optixDisplacedMicromesh/cmake_install.cmake")
  include("/home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/optixDynamicGeometry/cmake_install.cmake")
  include("/home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/optixDynamicMaterials/cmake_install.cmake")
  include("/home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/optixHair/cmake_install.cmake")
  include("/home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/optixHello/cmake_install.cmake")
  include("/home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/optixMeshViewer/cmake_install.cmake")
  include("/home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/optixModuleCreateAbort/cmake_install.cmake")
  include("/home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/optixMotionGeometry/cmake_install.cmake")
  include("/home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/optixMultiGPU/cmake_install.cmake")
  include("/home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/optixNVLink/cmake_install.cmake")
  include("/home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/optixOpticalFlow/cmake_install.cmake")
  include("/home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/optixOpacityMicromap/cmake_install.cmake")
  include("/home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/optixPathTracer/cmake_install.cmake")
  include("/home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/optixRaycasting/cmake_install.cmake")
  include("/home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/optixRibbons/cmake_install.cmake")
  include("/home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/optixSimpleMotionBlur/cmake_install.cmake")
  include("/home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/optixSphere/cmake_install.cmake")
  include("/home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/optixTriangle/cmake_install.cmake")
  include("/home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/optixVolumeViewer/cmake_install.cmake")
  include("/home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/optixWhitted/cmake_install.cmake")
  include("/home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/sutil/cmake_install.cmake")
  include("/home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/support/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
