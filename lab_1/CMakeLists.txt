add_library(matmul STATIC
  src/matrix.cc
  src/naive.cc
  src/transposed.cc
  src/blocked.cc
  src/strassen.cc
  src/simd.cc
  src/best.cc
  src/variants.cc
)
target_include_directories(matmul PUBLIC include)
target_link_libraries(matmul PUBLIC OpenMP::OpenMP_CXX)

# The scalar steps stay scalar so that the SIMD step shows its own effect.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(LAB1_NO_VECTORIZE -fno-vectorize -fno-slp-vectorize)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(LAB1_NO_VECTORIZE -fno-tree-vectorize)
endif()
set_source_files_properties(src/naive.cc src/transposed.cc src/blocked.cc
  PROPERTIES COMPILE_OPTIONS "${LAB1_NO_VECTORIZE}")

add_executable(bench bench/bench.cc)
target_link_libraries(bench PRIVATE matmul)

add_executable(matmul_test tests/matmul_test.cc)
target_link_libraries(matmul_test PRIVATE matmul)
add_test(NAME lab_1.matmul COMMAND matmul_test)
