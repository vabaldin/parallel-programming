add_library(matmul STATIC
  src/matrix.cpp
  src/naive.cpp
  src/transposed.cpp
  src/blocked.cpp
  src/strassen.cpp
  src/simd.cpp
  src/best.cpp
  src/variants.cpp
)
target_include_directories(matmul PUBLIC include)
target_link_libraries(matmul PUBLIC OpenMP::OpenMP_CXX)

# The scalar steps stay scalar so that the SIMD step shows its own effect.
# Loops are aligned because the short inner loop of blocked ran twice as slow
# in one copy whose body crossed a 64-byte boundary.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(LAB1_NO_VECTORIZE -fno-vectorize -fno-slp-vectorize)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(LAB1_NO_VECTORIZE -fno-tree-vectorize)
endif()
set_source_files_properties(src/naive.cpp src/transposed.cpp src/blocked.cpp
  PROPERTIES COMPILE_OPTIONS "${LAB1_NO_VECTORIZE};-falign-loops=64")

add_executable(bench bench/bench.cpp)
target_link_libraries(bench PRIVATE matmul)

add_executable(test_matmul tests/test_matmul.cpp)
target_link_libraries(test_matmul PRIVATE matmul)
add_test(NAME lab_1.matmul COMMAND test_matmul)
