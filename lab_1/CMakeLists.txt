add_library(matmul STATIC
  src/matrix.c
  src/naive.c
  src/transposed.c
  src/blocked.c
  src/strassen.c
  src/simd.c
  src/best.c
  src/variants.c
)
target_include_directories(matmul PUBLIC include)
target_link_libraries(matmul PUBLIC OpenMP::OpenMP_C m)

# The scalar steps stay scalar so that the SIMD step shows its own effect.
# Loops are aligned because the short inner loop of blocked ran twice as slow
# in one copy whose body crossed a 64-byte boundary.
if(CMAKE_C_COMPILER_ID MATCHES "Clang")
  set(LAB1_NO_VECTORIZE -fno-vectorize -fno-slp-vectorize)
elseif(CMAKE_C_COMPILER_ID STREQUAL "GNU")
  set(LAB1_NO_VECTORIZE -fno-tree-vectorize)
endif()
set_source_files_properties(src/naive.c src/transposed.c src/blocked.c
  PROPERTIES COMPILE_OPTIONS "${LAB1_NO_VECTORIZE};-falign-loops=64")

add_executable(bench bench/bench.c)
target_link_libraries(bench PRIVATE matmul)

add_executable(test_matmul tests/test_matmul.c)
target_link_libraries(test_matmul PRIVATE matmul)
add_test(NAME lab_1.matmul COMMAND test_matmul)
