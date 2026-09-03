# GenerateVulkanSpirvHeader.cmake
#
# Invokes glslangValidator to compile a Vulkan GLSL shader to SPIR-V and emit a
# C header embedding the SPIR-V words, then appends a *_count constant so the
# backend can bind the exact word count.
#
# Required variables:
#   GLSLANG_VALIDATOR  Absolute path to the glslangValidator binary.
#   INPUT_FILE         Absolute path to the .glsl source.
#   OUTPUT_FILE        Absolute path of the generated .spv.h header.
#   STAGE              One of: vert, tesc, tese, geom, frag, comp, mesh, task,
#                      rgen, rint, rahit, rchit, rmiss, rcall.
#   VARIABLE_NAME      C identifier for the embedded SPIR-V array.
#
# Optional variables:
#   TARGET_ENV         Vulkan target environment (default vulkan1.0).  Ray
#                      tracing shaders (GL_EXT_ray_tracing) require SPIR-V
#                      1.4+, so the RT entries pass vulkan1.2.
#
# Version pinning: the generated .spv.h headers are checked in, so the exact
# SPIR-V does not depend on the host glslangValidator.  Regeneration with a
# different glslangValidator version can emit different (though equivalent)
# SPIR-V and create diff churn on the vendored headers.  Pin a known-good
# version for the coin_regenerate_vulkan_spirv target and regenerate all
# headers in one run so they stay uniform; treat the check-in headers as the
# reference and only rebuild them on an intentional shader change.

if(NOT DEFINED GLSLANG_VALIDATOR)
  message(FATAL_ERROR "GLSLANG_VALIDATOR is not defined")
endif()
if(NOT EXISTS "${INPUT_FILE}")
  message(FATAL_ERROR "Input shader does not exist: ${INPUT_FILE}")
endif()
if(NOT DEFINED OUTPUT_FILE)
  message(FATAL_ERROR "OUTPUT_FILE is not defined")
endif()
if(NOT DEFINED VARIABLE_NAME)
  message(FATAL_ERROR "VARIABLE_NAME is not defined")
endif()
if(NOT DEFINED TARGET_ENV)
  set(TARGET_ENV vulkan1.0)
endif()

set(_tmp_output "${OUTPUT_FILE}.tmp")

# -V selects Vulkan SPIR-V output; -o ending in .h makes glslangValidator emit
# a C header of the form `const uint32_t <name>[] = { ... };`.
execute_process(
  COMMAND "${GLSLANG_VALIDATOR}" -V -S "${STAGE}" --target-env "${TARGET_ENV}"
          --variable-name "${VARIABLE_NAME}" -o "${_tmp_output}" "${INPUT_FILE}"
  RESULT_VARIABLE _result
  OUTPUT_VARIABLE _stdout
  ERROR_VARIABLE _stderr
)
if(NOT _result EQUAL 0)
  message(FATAL_ERROR
    "glslangValidator failed for ${INPUT_FILE} (stage ${STAGE}):\n${_stdout}\n${_stderr}")
endif()

# The generated header terminates with `};`; append the word-count constant.
file(APPEND "${_tmp_output}"
  "\nconst uint32_t ${VARIABLE_NAME}_count = sizeof(${VARIABLE_NAME}) / sizeof(uint32_t);\n")

file(RENAME "${_tmp_output}" "${OUTPUT_FILE}")
message(STATUS "Regenerated Vulkan SPIR-V header: ${OUTPUT_FILE}")
