cmake_minimum_required(VERSION 3.20)
project(parallel_programming LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(PP_NATIVE "Compile with -march=native" ON)
if(PP_NATIVE)
  add_compile_options(-march=native)
endif()
add_compile_options(-Wall -Wextra)
option(PP_WERROR "Treat warnings as errors" OFF)
if(PP_WERROR)
  add_compile_options(-Werror)
endif()

# Homebrew's libomp is keg-only; point FindOpenMP at it.
if(APPLE AND NOT OpenMP_ROOT)
  execute_process(COMMAND brew --prefix libomp
                  OUTPUT_VARIABLE OpenMP_ROOT
                  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
endif()
find_package(OpenMP REQUIRED)

enable_testing()

add_subdirectory(lab_1)
add_subdirectory(lab_2)
