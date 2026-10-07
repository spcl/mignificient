# Needs a GPU; run with: ctest -L integration --output-on-failure
add_test(NAME it_vector_add_bm
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/run.py --build ${CMAKE_BINARY_DIR}
          --case ${CMAKE_CURRENT_SOURCE_DIR}/cases/vector_add.json
          --executor bare-metal --signal iceoryx2)
set_tests_properties(it_vector_add_bm PROPERTIES LABELS "integration;gpu" TIMEOUT 900)
