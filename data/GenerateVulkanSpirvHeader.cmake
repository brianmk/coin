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
#   SPIRV_VAL          Absolute path to spirv-val.  When set, the compiled
#                      module is re-emitted as a temporary .spv and validated;
#                      invalid SPIR-V fails the regeneration.  Absent => the
#                      validation step is skipped (no build dependency).
#   SPIRV_OPT          Absolute path to spirv-opt (optional).  When set
#                      alongside SPIRV_VAL, the validated module is optimized
#                      with -O and the result is validated again.  This is a
#                      gate, not a code path: the committed .spv.h is still
#                      generated from the unoptimized compile so headers stay
#                      reproducible across toolchain versions.
#   GLSL_DEFINE        Optional single preprocessor define (e.g.
#                      COIN_ENABLE_DEBUG_PRINTF) applied to the compile.  Used
#                      to compile opt-in shader diagnostics without changing
#                      the committed headers (the define defaults off).
#   SPIRV_DIS          Absolute path to spirv-dis (optional).  When set
#                      alongside SPIRV_LAYOUT_CHECK, the validated module is
#                      reflected and its std140 block member offsets are
#                      asserted against the C++ struct mirrors.
#   SPIRV_LAYOUT_CHECK Absolute path to spirv_layout_check.py (optional).
#   SPIRV_LAYOUTS      Absolute path to the expected-layout JSON consumed by
#                      SPIRV_LAYOUT_CHECK.
#   PYTHON_EXECUTABLE  Interpreter used to run SPIRV_LAYOUT_CHECK.
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

set(_extra_args "")
if(DEFINED GLSL_DEFINE AND NOT GLSL_DEFINE STREQUAL "")
  # glslangValidator requires the define attached to the flag: -D<name>.
  list(APPEND _extra_args "-D${GLSL_DEFINE}")
endif()

set(_tmp_output "${OUTPUT_FILE}.tmp")

# -V selects Vulkan SPIR-V output; -o ending in .h makes glslangValidator emit
# a C header of the form `const uint32_t <name>[] = { ... };`.
execute_process(
  COMMAND "${GLSLANG_VALIDATOR}" -V -S "${STAGE}" --target-env "${TARGET_ENV}"
          ${_extra_args}
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

# Optional offline SPIR-V validation.  glslangValidator emits the C header
# directly, so validate by compiling the same source to a throwaway .spv.  The
# checked-in header is never produced from an optimized module (that would make
# it depend on the host spirv-opt version); spirv-opt is only used to prove the
# optimized module still validates.
if(DEFINED SPIRV_VAL AND NOT SPIRV_VAL STREQUAL "")
  set(_val_spv "${OUTPUT_FILE}.validate.spv")
  execute_process(
    COMMAND "${GLSLANG_VALIDATOR}" -V -S "${STAGE}" --target-env "${TARGET_ENV}"
            ${_extra_args}
            -o "${_val_spv}" "${INPUT_FILE}"
    RESULT_VARIABLE _val_compile_result
    OUTPUT_VARIABLE _val_compile_stdout
    ERROR_VARIABLE _val_compile_stderr
  )
  if(NOT _val_compile_result EQUAL 0)
    message(FATAL_ERROR
      "SPIR-V compile for validation failed for ${INPUT_FILE} (stage ${STAGE}):\n${_val_compile_stdout}\n${_val_compile_stderr}")
  endif()

  execute_process(
    COMMAND "${SPIRV_VAL}" --target-env "${TARGET_ENV}" "${_val_spv}"
    RESULT_VARIABLE _val_result
    OUTPUT_VARIABLE _val_stdout
    ERROR_VARIABLE _val_stderr
  )
  if(NOT _val_result EQUAL 0)
    message(FATAL_ERROR
      "spirv-val rejected ${INPUT_FILE} (stage ${STAGE}):\n${_val_stdout}\n${_val_stderr}")
  endif()

  if(DEFINED SPIRV_OPT AND NOT SPIRV_OPT STREQUAL "")
    set(_opt_spv "${OUTPUT_FILE}.optimized.spv")
    execute_process(
      COMMAND "${SPIRV_OPT}" -O "${_val_spv}" -o "${_opt_spv}"
      RESULT_VARIABLE _opt_result
      OUTPUT_VARIABLE _opt_stdout
      ERROR_VARIABLE _opt_stderr
    )
    if(NOT _opt_result EQUAL 0)
      message(FATAL_ERROR
        "spirv-opt failed for ${INPUT_FILE} (stage ${STAGE}):\n${_opt_stdout}\n${_opt_stderr}")
    endif()
    execute_process(
      COMMAND "${SPIRV_VAL}" --target-env "${TARGET_ENV}" "${_opt_spv}"
      RESULT_VARIABLE _opt_val_result
      OUTPUT_VARIABLE _opt_val_stdout
      ERROR_VARIABLE _opt_val_stderr
    )
    if(NOT _opt_val_result EQUAL 0)
      message(FATAL_ERROR
        "spirv-val rejected the optimized module for ${INPUT_FILE} (stage ${STAGE}):\n${_opt_val_stdout}\n${_opt_val_stderr}")
    endif()
    file(REMOVE "${_opt_spv}")
  endif()

  # Optional std140 layout reflection: assert the shader block member offsets
  # match the C++ struct mirrors (VulkanPushConstants, VulkanBackgroundPush,
  # VulkanLightingUbo, VulkanDrawUbo).  Uses spirv-dis rather than SPIRV-Reflect
  # (not packaged for these toolchains); see tools/rendering/spirv_layout_check.py.
  if(DEFINED SPIRV_LAYOUT_CHECK AND NOT SPIRV_LAYOUT_CHECK STREQUAL "" AND
     DEFINED SPIRV_DIS AND NOT SPIRV_DIS STREQUAL "")
    if(NOT DEFINED PYTHON_EXECUTABLE OR PYTHON_EXECUTABLE STREQUAL "")
      message(FATAL_ERROR
        "SPIRV_LAYOUT_CHECK is set but PYTHON_EXECUTABLE is not")
    endif()
    execute_process(
      COMMAND "${PYTHON_EXECUTABLE}" "${SPIRV_LAYOUT_CHECK}" "${_val_spv}"
              --layouts "${SPIRV_LAYOUTS}" --spirv-dis "${SPIRV_DIS}"
      RESULT_VARIABLE _refl_result
      OUTPUT_VARIABLE _refl_stdout
      ERROR_VARIABLE _refl_stderr
    )
    if(NOT _refl_result EQUAL 0)
      message(FATAL_ERROR
        "SPIR-V layout reflection failed for ${INPUT_FILE} (stage ${STAGE}):\n${_refl_stdout}\n${_refl_stderr}")
    endif()
    message(STATUS
      "Reflected Vulkan SPIR-V layouts: ${INPUT_FILE} (stage ${STAGE})")
  endif()

  file(REMOVE "${_val_spv}")
  message(STATUS "Validated Vulkan SPIR-V: ${INPUT_FILE} (stage ${STAGE})")
endif()

file(RENAME "${_tmp_output}" "${OUTPUT_FILE}")
message(STATUS "Regenerated Vulkan SPIR-V header: ${OUTPUT_FILE}")
