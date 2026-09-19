# CompareVulkanSpirvHeader.cmake
#
# Guard for the committed Vulkan SPIR-V headers.  Given a freshly regenerated
# header (GENERATED_FILE, produced by GenerateVulkanSpirvHeader.cmake) and the
# committed one (COMMITTED_FILE), fail with an actionable message when they
# differ.
#
# The comparison is a full byte comparison, so it catches edits to the shader
# source, to a same-directory #include, to the generator script, or to the
# compiler flags -- anything that would make the committed header stale.
#
# Required variables:
#   GENERATED_FILE  Header emitted by a fresh regeneration (build tree).
#   COMMITTED_FILE  Header checked into the source tree.
#   INPUT_FILE      Shader source, for the error message.

if(NOT EXISTS "${GENERATED_FILE}")
  message(FATAL_ERROR
    "SPIR-V check: the fresh regeneration did not produce ${GENERATED_FILE}")
endif()
if(NOT EXISTS "${COMMITTED_FILE}")
  message(FATAL_ERROR
    "SPIR-V check: the committed header is missing: ${COMMITTED_FILE}")
endif()

file(SHA256 "${GENERATED_FILE}" _generated_hash)
file(SHA256 "${COMMITTED_FILE}" _committed_hash)
if(_generated_hash STREQUAL _committed_hash)
  return()
endif()

# Fall back to a readable unified diff when the tool is available; the hash
# mismatch alone is enough to fail, this is only to make the fix obvious.
set(_diff_tool "")
find_program(_diff_tool diff)
if(_diff_tool)
  execute_process(
    COMMAND "${_diff_tool}" -u "${COMMITTED_FILE}" "${GENERATED_FILE}"
    OUTPUT_VARIABLE _diff_output
    ERROR_VARIABLE _diff_error)
  message(STATUS "SPIR-V header diff (${INPUT_FILE}):\n${_diff_output}")
endif()

message(FATAL_ERROR
  "Vulkan SPIR-V header is out of date:\n"
  "  committed: ${COMMITTED_FILE}\n"
  "  source:    ${INPUT_FILE}\n"
  "The committed header differs from a fresh regeneration. Run the\n"
  "coin_regenerate_vulkan_spirv target and commit the updated headers.")
