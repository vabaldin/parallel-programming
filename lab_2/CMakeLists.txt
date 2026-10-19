find_package(MPI REQUIRED COMPONENTS CXX)

add_library(lab2_loops STATIC
  src/grid.cc
  src/options.cc
  src/seq.cc
)
target_include_directories(lab2_loops PUBLIC include)

add_executable(lab2_seq src/run_seq.cc)
target_link_libraries(lab2_seq PRIVATE lab2_loops)

add_executable(lab2_omp src/run_omp.cc)
target_link_libraries(lab2_omp PRIVATE lab2_loops OpenMP::OpenMP_CXX)

add_executable(lab2_mpi src/run_mpi.cc)
target_link_libraries(lab2_mpi PRIVATE lab2_loops MPI::MPI_CXX)

# Odd sizes so that rows and columns do not split evenly; the tiny grid has
# more workers than rows.
set(lab2_sizes 103x77 4x9)
set(mpiexec ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG})

foreach(variant ref 2b 2b-cols 3b)
  foreach(threads 1 2 3 4 11)
    foreach(size IN LISTS lab2_sizes)
      string(REPLACE "x" ";" dims ${size})
      list(GET dims 0 ni)
      list(GET dims 1 nj)
      add_test(NAME lab_2.omp.${variant}.t${threads}.${size}
               COMMAND lab2_omp -v ${variant} -n ${ni} -m ${nj} -r 2
                       -t ${threads} -c)
    endforeach()
  endforeach()
endforeach()

foreach(variant ref 1b)
  foreach(np 1 2 3 4)
    foreach(size IN LISTS lab2_sizes)
      string(REPLACE "x" ";" dims ${size})
      list(GET dims 0 ni)
      list(GET dims 1 nj)
      add_test(NAME lab_2.mpi.${variant}.np${np}.${size}
               COMMAND ${mpiexec} ${np} ${MPIEXEC_PREFLAGS}
                       $<TARGET_FILE:lab2_mpi> ${MPIEXEC_POSTFLAGS}
                       -v ${variant} -n ${ni} -m ${nj} -r 2 -c)
    endforeach()
  endforeach()
endforeach()

set(compare_dir ${CMAKE_CURRENT_BINARY_DIR}/compare)
foreach(spec "omp;ref;ref" "omp;2b;2b" "omp;2b-cols;2b" "omp;3b;3b"
             "mpi;ref;ref" "mpi;1b;1b")
  list(GET spec 0 impl)
  list(GET spec 1 variant)
  list(GET spec 2 seq_variant)
  if(impl STREQUAL "mpi")
    set(launcher "${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 3")
  else()
    set(launcher "")
  endif()
  add_test(NAME lab_2.files.${impl}.${variant}
           COMMAND ${CMAKE_COMMAND}
                   -DSEQ=$<TARGET_FILE:lab2_seq> -DPAR=$<TARGET_FILE:lab2_${impl}>
                   -DLAUNCHER=${launcher} -DVARIANT=${variant}
                   -DSEQ_VARIANT=${seq_variant} -DNI=301 -DNJ=211
                   -DWORKDIR=${compare_dir}/${impl}_${variant}
                   -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/compare_files.cmake)
  if(impl STREQUAL "omp")
    set_tests_properties(lab_2.files.${impl}.${variant} PROPERTIES
                         ENVIRONMENT OMP_NUM_THREADS=3)
  endif()
endforeach()
