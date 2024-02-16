//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#pragma once

#define SAMPLES_DIR "/home/fzaghloul/workspace/cs2240/path2-fmz/SDK"
#define SAMPLES_PTX_DIR "/home/fzaghloul/workspace/cs2240/path2-fmz/SDK/build-SDK-Desktop_Qt_6_5_2_GCC_64bit-Debug/lib/ptx"
#define SAMPLES_CUDA_DIR "/home/fzaghloul/workspace/cs2240/path2-fmz/SDK/cuda"

// Include directories
#define SAMPLES_RELATIVE_INCLUDE_DIRS \
  "cuda", \
  "sutil", \
  ".", 
#define SAMPLES_ABSOLUTE_INCLUDE_DIRS \
  "/home/fzaghloul/workspace/cs2240/path2-fmz/include", \
  "/usr/local/cuda/include", 

// Signal whether to use NVRTC or not
#define CUDA_NVRTC_ENABLED 0

// NVRTC compiler options
#if defined( NDEBUG )
#define CUDA_NVRTC_OPTIONS  \
  "-arch", \
  "compute_70", \
  "-G", \
  "-use_fast_math", \
  "-default-device", \
  "-rdc", \
  "true", \
  "-D__x86_64",
#else
#define CUDA_NVRTC_OPTIONS  \
  "-arch", \
  "compute_70", \
  "-G", \
  "-use_fast_math", \
  "-default-device", \
  "-rdc", \
  "true", \
  "-D__x86_64",
#endif

// Indicate what input we are generating
#define SAMPLES_INPUT_GENERATE_OPTIXIR 1
#define SAMPLES_INPUT_GENERATE_PTX 0
