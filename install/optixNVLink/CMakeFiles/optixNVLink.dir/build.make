# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /snap/cmake/1366/bin/cmake

# The command to remove a file.
RM = /snap/cmake/1366/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/SDK

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install

# Include any dependencies generated for this target.
include optixNVLink/CMakeFiles/optixNVLink.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include optixNVLink/CMakeFiles/optixNVLink.dir/compiler_depend.make

# Include the progress variables for this target.
include optixNVLink/CMakeFiles/optixNVLink.dir/progress.make

# Include the compile flags for this target's objects.
include optixNVLink/CMakeFiles/optixNVLink.dir/flags.make

lib/ptx/optixNVLink_generated_optixNVLink_kernels.cu.o: /home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/SDK/optixNVLink/optixNVLink_kernels.cu
lib/ptx/optixNVLink_generated_optixNVLink_kernels.cu.o: optixNVLink/CMakeFiles/optixNVLink.dir/optixNVLink_generated_optixNVLink_kernels.cu.o.depend
lib/ptx/optixNVLink_generated_optixNVLink_kernels.cu.o: optixNVLink/CMakeFiles/optixNVLink.dir/optixNVLink_generated_optixNVLink_kernels.cu.o.cmake
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object lib/ptx/optixNVLink_generated_optixNVLink_kernels.cu.o"
	cd /home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/optixNVLink/CMakeFiles/optixNVLink.dir && /snap/cmake/1366/bin/cmake -E make_directory /home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/lib/ptx/.
	cd /home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/optixNVLink/CMakeFiles/optixNVLink.dir && /snap/cmake/1366/bin/cmake -D verbose:BOOL=$(VERBOSE) -D check_dependencies:BOOL=OFF -D build_configuration:STRING=Release -D generated_file:STRING=/home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/lib/ptx/./optixNVLink_generated_optixNVLink_kernels.cu.o -D generated_cubin_file:STRING=/home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/lib/ptx/./optixNVLink_generated_optixNVLink_kernels.cu.o.cubin.txt -D generated_fatbin_file:STRING=/home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/lib/ptx/./optixNVLink_generated_optixNVLink_kernels.cu.o.fatbin.txt -P /home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/optixNVLink/CMakeFiles/optixNVLink.dir//optixNVLink_generated_optixNVLink_kernels.cu.o.cmake

lib/ptx/optixNVLink_generated_optixNVLink.cu.optixir: /home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/SDK/optixNVLink/optixNVLink.cu
lib/ptx/optixNVLink_generated_optixNVLink.cu.optixir: optixNVLink/CMakeFiles/optixNVLink.dir/optixNVLink_generated_optixNVLink.cu.optixir.depend
lib/ptx/optixNVLink_generated_optixNVLink.cu.optixir: optixNVLink/CMakeFiles/optixNVLink.dir/optixNVLink_generated_optixNVLink.cu.optixir.cmake
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building NVCC optixir file lib/ptx/optixNVLink_generated_optixNVLink.cu.optixir"
	cd /home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/optixNVLink/CMakeFiles/optixNVLink.dir && /snap/cmake/1366/bin/cmake -E make_directory /home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/lib/ptx/.
	cd /home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/optixNVLink/CMakeFiles/optixNVLink.dir && /snap/cmake/1366/bin/cmake -D verbose:BOOL=$(VERBOSE) -D check_dependencies:BOOL=OFF -D build_configuration:STRING=Release -D generated_file:STRING=/home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/lib/ptx/./optixNVLink_generated_optixNVLink.cu.optixir -D generated_cubin_file:STRING=/home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/lib/ptx/./optixNVLink_generated_optixNVLink.cu.optixir.cubin.txt -D generated_fatbin_file:STRING=/home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/lib/ptx/./optixNVLink_generated_optixNVLink.cu.optixir.fatbin.txt -P /home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/optixNVLink/CMakeFiles/optixNVLink.dir//optixNVLink_generated_optixNVLink.cu.optixir.cmake

optixNVLink/CMakeFiles/optixNVLink.dir/optixNVLink.cpp.o: optixNVLink/CMakeFiles/optixNVLink.dir/flags.make
optixNVLink/CMakeFiles/optixNVLink.dir/optixNVLink.cpp.o: /home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/SDK/optixNVLink/optixNVLink.cpp
optixNVLink/CMakeFiles/optixNVLink.dir/optixNVLink.cpp.o: optixNVLink/CMakeFiles/optixNVLink.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object optixNVLink/CMakeFiles/optixNVLink.dir/optixNVLink.cpp.o"
	cd /home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/optixNVLink && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT optixNVLink/CMakeFiles/optixNVLink.dir/optixNVLink.cpp.o -MF CMakeFiles/optixNVLink.dir/optixNVLink.cpp.o.d -o CMakeFiles/optixNVLink.dir/optixNVLink.cpp.o -c /home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/SDK/optixNVLink/optixNVLink.cpp

optixNVLink/CMakeFiles/optixNVLink.dir/optixNVLink.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/optixNVLink.dir/optixNVLink.cpp.i"
	cd /home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/optixNVLink && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/SDK/optixNVLink/optixNVLink.cpp > CMakeFiles/optixNVLink.dir/optixNVLink.cpp.i

optixNVLink/CMakeFiles/optixNVLink.dir/optixNVLink.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/optixNVLink.dir/optixNVLink.cpp.s"
	cd /home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/optixNVLink && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/SDK/optixNVLink/optixNVLink.cpp -o CMakeFiles/optixNVLink.dir/optixNVLink.cpp.s

# Object files for target optixNVLink
optixNVLink_OBJECTS = \
"CMakeFiles/optixNVLink.dir/optixNVLink.cpp.o"

# External object files for target optixNVLink
optixNVLink_EXTERNAL_OBJECTS = \
"/home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/lib/ptx/optixNVLink_generated_optixNVLink_kernels.cu.o"

bin/optixNVLink: optixNVLink/CMakeFiles/optixNVLink.dir/optixNVLink.cpp.o
bin/optixNVLink: lib/ptx/optixNVLink_generated_optixNVLink_kernels.cu.o
bin/optixNVLink: optixNVLink/CMakeFiles/optixNVLink.dir/build.make
bin/optixNVLink: lib/libimgui.a
bin/optixNVLink: lib/libsutil_7_sdk.so
bin/optixNVLink: /usr/local/cuda/lib64/libcudart_static.a
bin/optixNVLink: /usr/lib/x86_64-linux-gnu/librt.so
bin/optixNVLink: /usr/lib/x86_64-linux-gnu/libdl.so
bin/optixNVLink: lib/libglfw.so.3.2
bin/optixNVLink: lib/libglad.so
bin/optixNVLink: /usr/lib/x86_64-linux-gnu/libOpenGL.so
bin/optixNVLink: /usr/lib/x86_64-linux-gnu/libGLX.so
bin/optixNVLink: optixNVLink/CMakeFiles/optixNVLink.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable ../bin/optixNVLink"
	cd /home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/optixNVLink && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/optixNVLink.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
optixNVLink/CMakeFiles/optixNVLink.dir/build: bin/optixNVLink
.PHONY : optixNVLink/CMakeFiles/optixNVLink.dir/build

optixNVLink/CMakeFiles/optixNVLink.dir/clean:
	cd /home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/optixNVLink && $(CMAKE_COMMAND) -P CMakeFiles/optixNVLink.dir/cmake_clean.cmake
.PHONY : optixNVLink/CMakeFiles/optixNVLink.dir/clean

optixNVLink/CMakeFiles/optixNVLink.dir/depend: lib/ptx/optixNVLink_generated_optixNVLink.cu.optixir
optixNVLink/CMakeFiles/optixNVLink.dir/depend: lib/ptx/optixNVLink_generated_optixNVLink_kernels.cu.o
	cd /home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/SDK /home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/SDK/optixNVLink /home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install /home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/optixNVLink /home/fzaghloul/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/install/optixNVLink/CMakeFiles/optixNVLink.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : optixNVLink/CMakeFiles/optixNVLink.dir/depend

