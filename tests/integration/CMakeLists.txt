# Needs a GPU; run with: ctest -L integration --output-on-failure
add_test(NAME it_vector_add_bm
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/run.py --build ${CMAKE_BINARY_DIR}
          --case ${CMAKE_CURRENT_SOURCE_DIR}/cases/vector_add.json
          --executor bare-metal --signal iceoryx2)
set_tests_properties(it_vector_add_bm PROPERTIES LABELS "integration;gpu" TIMEOUT 900)

# ResNet-50 from a python package (tools/pack_code.py: conda env, weights, cubin analysis), packed once into
# <build>/test-packages and reused until its sources change. Packing needs network and conda; set
# CONDA_PKGS_DIRS to a directory on the build's filesystem to hard-link the env.
add_test(NAME it_pack_resnet50
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/pack.py ${CMAKE_BINARY_DIR}
          ${CMAKE_CURRENT_SOURCE_DIR}/functions/resnet50 python)
set_tests_properties(it_pack_resnet50 PROPERTIES LABELS "integration" TIMEOUT 1800 FIXTURES_SETUP resnet50_pkg)
foreach(executor bare-metal container)
  string(REPLACE "bare-metal" "bm" short ${executor})
  string(REPLACE "container" "ct" short ${short})
  add_test(NAME it_resnet50_${short}
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/run.py --build ${CMAKE_BINARY_DIR}
            --case ${CMAKE_CURRENT_SOURCE_DIR}/cases/resnet50.json
            --executor ${executor} --signal iceoryx2)
  set_tests_properties(it_resnet50_${short} PROPERTIES LABELS "integration;gpu" TIMEOUT 900
                       FIXTURES_REQUIRED resnet50_pkg)
endforeach()
