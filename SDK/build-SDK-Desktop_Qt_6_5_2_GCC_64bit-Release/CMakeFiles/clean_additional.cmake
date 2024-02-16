# Additional clean files
cmake_minimum_required(VERSION 3.16)

if("${CONFIG}" STREQUAL "" OR "${CONFIG}" STREQUAL "Release")
  file(REMOVE_RECURSE
  "path-fmz/CMakeFiles/nvpath-fmz_autogen.dir/AutogenUsed.txt"
  "path-fmz/CMakeFiles/nvpath-fmz_autogen.dir/ParseCache.txt"
  "path-fmz/nvpath-fmz_autogen"
  )
endif()
