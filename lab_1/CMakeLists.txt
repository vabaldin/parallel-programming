add_library(matmul STATIC
  src/matrix.cc
  src/naive.cc
  src/variants.cc
)
target_include_directories(matmul PUBLIC include)
target_link_libraries(matmul PUBLIC OpenMP::OpenMP_CXX)

add_executable(bench bench/bench.cc)
target_link_libraries(bench PRIVATE matmul)

add_executable(matmul_test tests/matmul_test.cc)
target_link_libraries(matmul_test PRIVATE matmul)
add_test(NAME lab_1.matmul COMMAND matmul_test)
