if(NOT DEFINED COIN_SOURCE_DIR OR NOT DEFINED COIN_BINARY_DIR)
  message(FATAL_ERROR "ShaderGeneratorTest requires source and binary directories")
endif()

set(_fixture_dir "${COIN_BINARY_DIR}/shader-generator-fixture")
file(REMOVE_RECURSE "${_fixture_dir}")
file(MAKE_DIRECTORY
  "${_fixture_dir}/one"
  "${_fixture_dir}/two"
  "${_fixture_dir}/nested")

set(_root "${_fixture_dir}/Root.glsl")
set(_one "${_fixture_dir}/one/Common.glsl")
set(_two "${_fixture_dir}/two/Common.glsl")
set(_nested "${_fixture_dir}/nested/Shared.glsl")
set(_output "${_fixture_dir}/Root.h")
set(_depfile "${_fixture_dir}/Root.h.d")
set(_template "${COIN_SOURCE_DIR}/data/strfytemplate.cmake.in")
set(_shader_generator "${COIN_SOURCE_DIR}/data/GenerateShaderHeader.cmake")
set(_text_generator "${COIN_SOURCE_DIR}/data/GenerateTextHeader.cmake")

file(WRITE "${_root}" "#version 410 core\n// #include \"does-not-exist.glsl\"\n/*\n#include \"does-not-exist-in-a-block-comment.glsl\"\n*/\n#include \"one/Common.glsl\"\n#include \"two/Common.glsl\"\nvoid main() { gl_Position = vec4(coin_one(), coin_two(), 0.0, 1.0); }\n")
file(WRITE "${_one}" "#include \"../nested/Shared.glsl\"\nfloat coin_one() { return coin_shared(); }\n")
file(WRITE "${_two}" "float coin_two() { return 1.0; }\n")
file(WRITE "${_nested}" "float coin_shared() { return 1.0; }\n")

set(_shader_command
  "${CMAKE_COMMAND}"
  "-DINPUT_FILE=${_root}"
  "-DOUTPUT_FILE=${_output}"
  "-DDEPFILE=${_depfile}"
  "-DSOURCE_ROOT=${_fixture_dir}"
  "-DTEMPLATE_FILE=${_template}"
  "-DCOIN_HEADER_DEF=COIN_SHADER_TEST_ROOT_H"
  "-DCOIN_TEXTVAR_NAME=coin_test_root_shadersource"
  -P "${_shader_generator}")
execute_process(COMMAND ${_shader_command}
  RESULT_VARIABLE _result
  OUTPUT_VARIABLE _stdout
  ERROR_VARIABLE _stderr)
if(NOT _result EQUAL 0)
  message(FATAL_ERROR "valid shader generation failed: ${_stdout}${_stderr}")
endif()

file(READ "${_output}" _generated)
file(READ "${_depfile}" _dependencies)
foreach(_expected IN ITEMS "${_root}" "${_one}" "${_two}" "${_nested}")
  string(FIND "${_dependencies}" "${_expected}" _dependency_index)
  if(_dependency_index EQUAL -1)
    message(FATAL_ERROR "depfile omitted transitive dependency: ${_expected}")
  endif()
endforeach()
string(FIND "${_generated}" "#line 1" _line_index)
if(_line_index EQUAL -1)
  message(FATAL_ERROR "generated shader omitted source mapping")
endif()
string(FIND "${_generated}" "coin-source: nested/Shared.glsl" _source_index)
if(_source_index EQUAL -1)
  message(FATAL_ERROR "generated shader omitted nested source identity")
endif()
string(FIND "${_generated}" "coin-source-id: 1 one/Common.glsl" _source_id_index)
if(_source_id_index EQUAL -1)
  message(FATAL_ERROR "generated shader omitted source ID mapping")
endif()

file(WRITE "${_two}" "float coin_two() { return 2.0; }\n")
execute_process(COMMAND ${_shader_command}
  RESULT_VARIABLE _result
  ERROR_VARIABLE _stderr)
if(NOT _result EQUAL 0)
  message(FATAL_ERROR "shader regeneration failed: ${_stderr}")
endif()
file(READ "${_output}" _regenerated)
if(_generated STREQUAL _regenerated)
  message(FATAL_ERROR "editing an included module did not regenerate the root header")
endif()

set(_depfile_project "${_fixture_dir}/depfile-project")
set(_depfile_build "${_fixture_dir}/depfile-build")
set(_depfile_timestamp_marker "${_depfile_build}/timestamp-marker")
file(MAKE_DIRECTORY "${_depfile_project}")
file(WRITE "${_depfile_project}/Root.glsl"
  "#version 410 core\n#include \"Module.glsl\"\nvoid main() { float value = coin_value(); }\n")
file(WRITE "${_depfile_project}/Module.glsl"
  "float coin_value() { return 1.0; }\n")
file(WRITE "${_depfile_project}/Other.glsl"
  "float coin_other() { return 3.0; }\n")
set(_depfile_cmake [=[
cmake_minimum_required(VERSION 3.21)
project(CoinShaderDepfileFixture NONE)

set(_generator "__COIN_SOURCE_DIR__/data/GenerateShaderHeader.cmake")
set(_template "__COIN_SOURCE_DIR__/data/strfytemplate.cmake.in")
set(_root "${CMAKE_CURRENT_SOURCE_DIR}/Root.glsl")
set(_output "${CMAKE_CURRENT_BINARY_DIR}/Root.h")

add_custom_command(
  OUTPUT "${_output}"
  BYPRODUCTS "${_output}.d"
  COMMAND "${CMAKE_COMMAND}"
    "-DINPUT_FILE=${_root}"
    "-DOUTPUT_FILE=${_output}"
    "-DDEPFILE=${_output}.d"
    "-DSOURCE_ROOT=${CMAKE_CURRENT_SOURCE_DIR}"
    "-DTEMPLATE_FILE=${_template}"
    "-DCOIN_HEADER_DEF=COIN_SHADER_DEPFILE_ROOT_H"
    "-DCOIN_TEXTVAR_NAME=coin_shader_depfile_root"
    -P "${_generator}"
  DEPENDS
    "${_root}"
    "${_generator}"
    "${_template}"
  DEPFILE "${_output}.d"
  VERBATIM
)
add_custom_target(shader ALL DEPENDS "${_output}")
]=])
string(REPLACE "__COIN_SOURCE_DIR__" "${COIN_SOURCE_DIR}"
  _depfile_cmake "${_depfile_cmake}")
file(WRITE "${_depfile_project}/CMakeLists.txt" "${_depfile_cmake}")

set(_depfile_configure_command
  "${CMAKE_COMMAND}" -S "${_depfile_project}" -B "${_depfile_build}")
if(DEFINED COIN_GENERATOR AND NOT COIN_GENERATOR STREQUAL "")
  list(APPEND _depfile_configure_command -G "${COIN_GENERATOR}")
endif()
execute_process(COMMAND ${_depfile_configure_command}
  RESULT_VARIABLE _result
  OUTPUT_VARIABLE _stdout
  ERROR_VARIABLE _stderr)
if(NOT _result EQUAL 0)
  message(FATAL_ERROR "depfile fixture configure failed: ${_stdout}${_stderr}")
endif()

function(run_depfile_fixture_build _step)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" --build "${_depfile_build}"
    RESULT_VARIABLE _build_result
    OUTPUT_VARIABLE _build_stdout
    ERROR_VARIABLE _build_stderr)
  if(NOT _build_result EQUAL 0)
    message(FATAL_ERROR
      "depfile fixture build failed (${_step}): ${_build_stdout}${_build_stderr}")
  endif()
endfunction()

# Make source edits strictly newer than the generated output.  Some platforms
# and Make implementations expose only coarse timestamp resolution, so an
# immediate rewrite can otherwise be missed even when the depfile is correct.
function(wait_for_depfile_timestamp _reference _marker)
  file(TIMESTAMP "${_reference}" _reference_timestamp "%s")
  if(NOT _reference_timestamp)
    message(FATAL_ERROR
      "cannot read timestamp for depfile fixture output: ${_reference}")
  endif()
  while(1)
    file(TOUCH "${_marker}")
    file(TIMESTAMP "${_marker}" _marker_timestamp "%s")
    if(_marker_timestamp GREATER _reference_timestamp)
      break()
    endif()
    execute_process(COMMAND "${CMAKE_COMMAND}" -E sleep 1)
  endwhile()
endfunction()

run_depfile_fixture_build(initial)
file(READ "${_depfile_build}/Root.h" _depfile_initial)

wait_for_depfile_timestamp("${_depfile_build}/Root.h"
  "${_depfile_timestamp_marker}")
file(WRITE "${_depfile_project}/Module.glsl"
  "float coin_value() { return 2.0; }\n")
run_depfile_fixture_build(included-module-edit)
file(READ "${_depfile_build}/Root.h" _depfile_module_edit)
if(_depfile_initial STREQUAL _depfile_module_edit)
  message(FATAL_ERROR
    "build-system depfile did not rebuild after an included module changed")
endif()

wait_for_depfile_timestamp("${_depfile_build}/Root.h"
  "${_depfile_timestamp_marker}")
file(WRITE "${_depfile_project}/Root.glsl"
  "#version 410 core\n#include \"Other.glsl\"\nvoid main() { float value = coin_other(); }\n")
run_depfile_fixture_build(include-topology-edit)
file(READ "${_depfile_build}/Root.h" _depfile_topology_edit)
if(NOT _depfile_topology_edit MATCHES "coin_other")
  message(FATAL_ERROR
    "build-system depfile fixture did not follow an include topology change")
endif()

wait_for_depfile_timestamp("${_depfile_build}/Root.h"
  "${_depfile_timestamp_marker}")
file(WRITE "${_depfile_project}/Other.glsl"
  "float coin_other() { return 4.0; }\n")
run_depfile_fixture_build(new-included-module-edit)
file(READ "${_depfile_build}/Root.h" _depfile_new_module_edit)
if(_depfile_topology_edit STREQUAL _depfile_new_module_edit)
  message(FATAL_ERROR
    "build-system depfile did not track the newly included module")
endif()

set(_ordinary "${_fixture_dir}/ordinary.iv")
set(_ordinary_output "${_fixture_dir}/ordinary.h")
file(WRITE "${_ordinary}" "Separator { label \"ordinary\\\\resource\" }\n")
execute_process(
  COMMAND "${CMAKE_COMMAND}"
    "-DINPUT_FILE=${_ordinary}"
    "-DOUTPUT_FILE=${_ordinary_output}"
    "-DTEMPLATE_FILE=${_template}"
    -DCOIN_HEADER_DEF=COIN_ORDINARY_RESOURCE_H
    -DCOIN_TEXTVAR_NAME=ordinary_resource
    -P "${_text_generator}"
  RESULT_VARIABLE _result
  ERROR_VARIABLE _stderr)
if(NOT _result EQUAL 0)
  message(FATAL_ERROR "ordinary text generation failed: ${_stderr}")
endif()
file(READ "${_ordinary_output}" _ordinary_header)
if(_ordinary_header MATCHES "#line|coin-source")
  message(FATAL_ERROR "ordinary text generation applied shader preprocessing")
endif()
if(NOT _ordinary_header MATCHES "ordinary.*resource")
  message(FATAL_ERROR "ordinary text escaping changed unexpectedly")
endif()

function(expect_shader_failure _name _source _message)
  set(_failure_root "${_fixture_dir}/${_name}.glsl")
  set(_failure_output "${_fixture_dir}/${_name}.h")
  set(_failure_depfile "${_fixture_dir}/${_name}.h.d")
  file(WRITE "${_failure_root}" "${_source}")
  execute_process(
    COMMAND "${CMAKE_COMMAND}"
      "-DINPUT_FILE=${_failure_root}"
      "-DOUTPUT_FILE=${_failure_output}"
      "-DDEPFILE=${_failure_depfile}"
      "-DSOURCE_ROOT=${_fixture_dir}"
      "-DTEMPLATE_FILE=${_template}"
      -DCOIN_HEADER_DEF=COIN_SHADER_FAILURE_H
      -DCOIN_TEXTVAR_NAME=coin_shader_failure
      -P "${_shader_generator}"
    RESULT_VARIABLE _failure_result
    ERROR_VARIABLE _failure_stderr)
  if(_failure_result EQUAL 0)
    message(FATAL_ERROR "${_message}")
  endif()
endfunction()

expect_shader_failure(
  missing
  "#version 410 core\n#include \"does-not-exist.glsl\"\nvoid main() {}\n"
  "missing include unexpectedly succeeded")
file(WRITE "${_fixture_dir}/cycle-a.glsl" "#include \"cycle-b.glsl\"\n")
file(WRITE "${_fixture_dir}/cycle-b.glsl" "#include \"cycle-a.glsl\"\n")
expect_shader_failure(
  cycle
  "#version 410 core\n#include \"cycle-a.glsl\"\nvoid main() {}\n"
  "include cycle unexpectedly succeeded")
file(WRITE "${_fixture_dir}/version-module.glsl" "#version 410 core\nfloat value() { return 1.0; }\n")
expect_shader_failure(
  module-version
  "#version 410 core\n#include \"version-module.glsl\"\nvoid main() {}\n"
  "module #version unexpectedly succeeded")
