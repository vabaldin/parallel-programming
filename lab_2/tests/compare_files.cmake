# Runs the sequential and a parallel program with -o and requires the two
# result files to be identical, as the manual prescribes.
# Variables: SEQ, PAR, LAUNCHER (space-separated, may be empty), VARIANT,
# SEQ_VARIANT, NI, NJ, WORKDIR.
file(MAKE_DIRECTORY "${WORKDIR}")
set(seq_out "${WORKDIR}/seq_${VARIANT}.txt")
set(par_out "${WORKDIR}/par_${VARIANT}.txt")
file(REMOVE "${seq_out}" "${par_out}")
separate_arguments(launcher UNIX_COMMAND "${LAUNCHER}")

execute_process(
  COMMAND "${SEQ}" -v "${SEQ_VARIANT}" -n "${NI}" -m "${NJ}" -r 1 -o "${seq_out}"
  RESULT_VARIABLE rc)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "sequential run failed: ${rc}")
endif()

execute_process(
  COMMAND ${launcher} "${PAR}" -v "${VARIANT}" -n "${NI}" -m "${NJ}" -r 1 -o "${par_out}"
  RESULT_VARIABLE rc)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "parallel run failed: ${rc}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E compare_files "${seq_out}" "${par_out}"
  RESULT_VARIABLE rc)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "${seq_out} and ${par_out} differ")
endif()
message(STATUS "${VARIANT}: result files are identical")
