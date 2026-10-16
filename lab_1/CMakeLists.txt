add_library(matmul STATIC
  src/matrix.c
  src/naive.c
  src/variants.c
)
target_include_directories(matmul PUBLIC include)
target_link_libraries(matmul PUBLIC OpenMP::OpenMP_C m)

add_executable(bench bench/bench.c)
target_link_libraries(bench PRIVATE matmul)

add_executable(test_matmul tests/test_matmul.c)
target_link_libraries(test_matmul PRIVATE matmul)
add_test(NAME lab_1.matmul COMMAND test_matmul)
