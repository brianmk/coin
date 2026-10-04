# CoinCompileSpirvEmbed.cmake
#
# Build-time helper (invoked with `cmake -P`): compile a single Vulkan GLSL
# shader to SPIR-V with glslangValidator and embed the result as a generated C
# header.  The build rules that call this live in CoinCompileSpirv.cmake.
#
# Required -D arguments:
#   COIN_GLSLANG_VALIDATOR  path to glslangValidator
#   COIN_SHADER_SOURCE      absolute path to the .glsl source
#   COIN_SHADER_STAGE       vert | frag | comp | ...
#   COIN_SHADER_HEADER      absolute path of the .spv.h to generate
#   COIN_SHADER_VARIABLE    C symbol for the embedded uint32_t array
#   COIN_SHADER_SOURCE_REL  source path recorded in the header banner

if(NOT COIN_GLSLANG_VALIDATOR OR NOT COIN_SHADER_SOURCE OR NOT COIN_SHADER_STAGE
   OR NOT COIN_SHADER_HEADER OR NOT COIN_SHADER_VARIABLE
   OR NOT COIN_SHADER_SOURCE_REL)
  message(FATAL_ERROR "CoinCompileSpirvEmbed.cmake: missing a required -D argument")
endif()

get_filename_component(_outdir "${COIN_SHADER_HEADER}" DIRECTORY)
file(MAKE_DIRECTORY "${_outdir}")

set(_spv "${COIN_SHADER_HEADER}.spv")

execute_process(
  COMMAND "${COIN_GLSLANG_VALIDATOR}" -V -S "${COIN_SHADER_STAGE}"
          -o "${_spv}" "${COIN_SHADER_SOURCE}"
  RESULT_VARIABLE _result
  OUTPUT_VARIABLE _log
  ERROR_VARIABLE _errlog)
if(NOT _result EQUAL 0)
  message(FATAL_ERROR
    "glslangValidator failed for ${COIN_SHADER_SOURCE_REL}:\n${_log}${_errlog}")
endif()

execute_process(
  COMMAND "${COIN_GLSLANG_VALIDATOR}" --version
  OUTPUT_VARIABLE _version_raw ERROR_VARIABLE _version_raw)
string(REGEX MATCH "Glslang Version: [0-9:.]+" _version "${_version_raw}")
string(REPLACE "Glslang Version: " "" _version "${_version}")
if(_version STREQUAL "")
  set(_version "unknown")
endif()

file(READ "${_spv}" _hex HEX)
file(REMOVE "${_spv}")

string(LENGTH "${_hex}" _hexlen)
math(EXPR _words "${_hexlen} / 8")

# Format the little-endian 32-bit words, eight per line, to match the format
# of the headers this replaced.
set(_body "")
set(_line "")
set(_i 0)
while(_i LESS _words)
  math(EXPR _off "${_i} * 8")
  string(SUBSTRING "${_hex}" ${_off} 8 _word)
  string(SUBSTRING "${_word}" 0 2 _b0)
  string(SUBSTRING "${_word}" 2 2 _b1)
  string(SUBSTRING "${_word}" 4 2 _b2)
  string(SUBSTRING "${_word}" 6 2 _b3)
  set(_line "${_line}0x${_b3}${_b2}${_b1}${_b0},")
  math(EXPR _i "${_i} + 1")
  math(EXPR _rem "${_i} % 8")
  if(_rem EQUAL 0)
    set(_body "${_body}\t${_line}\n")
    set(_line "")
  endif()
endwhile()
if(NOT _line STREQUAL "")
  set(_body "${_body}\t${_line}\n")
endif()

file(WRITE "${COIN_SHADER_HEADER}"
"// Generated from ${COIN_SHADER_SOURCE_REL} with glslangValidator ${_version}.
// Do not edit by hand; regenerate it from that source.
#pragma once
#include <cstdint>
const uint32_t ${COIN_SHADER_VARIABLE}[] = {
${_body}};
const uint32_t ${COIN_SHADER_VARIABLE}_count = sizeof(${COIN_SHADER_VARIABLE}) / sizeof(uint32_t);
")
