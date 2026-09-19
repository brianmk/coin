macro(coin_embed_text _input _postfix)
  get_filename_component(_basename "${_input}" NAME_WE)
  get_filename_component(_suffix "${_input}" EXT)
  get_filename_component(_path "${_input}" PATH)
  string(SUBSTRING "${_suffix}" 1 -1 _suffix_name)
  string(TOUPPER "SO_${_basename}_${_suffix_name}_H" COIN_HEADER_DEF)
  string(TOUPPER "${_basename}" COIN_TEXTVAR_NAME)
  set(COIN_TEXTVAR_NAME "${COIN_TEXTVAR_NAME}${_postfix}")

  set(_input_file "${PROJECT_SOURCE_DIR}/${_input}")
  set(_output_file "${CMAKE_CURRENT_BINARY_DIR}/${_path}/${_basename}.h")
  add_custom_command(
    OUTPUT "${_output_file}"
    COMMAND "${CMAKE_COMMAND}"
      "-DINPUT_FILE=${_input_file}"
      "-DOUTPUT_FILE=${_output_file}"
      "-DTEMPLATE_FILE=${CMAKE_CURRENT_SOURCE_DIR}/strfytemplate.cmake.in"
      "-DCOIN_HEADER_DEF=${COIN_HEADER_DEF}"
      "-DCOIN_TEXTVAR_NAME=${COIN_TEXTVAR_NAME}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/GenerateTextHeader.cmake"
    DEPENDS
      "${_input_file}"
      "${CMAKE_CURRENT_SOURCE_DIR}/GenerateTextHeader.cmake"
      "${CMAKE_CURRENT_SOURCE_DIR}/strfytemplate.cmake.in"
    VERBATIM
  )
  list(APPEND COIN_GENERATED_DATA_HEADERS "${_output_file}")
endmacro()

macro(coin_embed_gl_shader _name _input)
  set(_input_file "${PROJECT_SOURCE_DIR}/${_input}")
  if(NOT EXISTS "${_input_file}")
    message(FATAL_ERROR "GLSL root shader does not exist: ${_input}")
  endif()

  file(RELATIVE_PATH _source_relative "${PROJECT_SOURCE_DIR}" "${_input_file}")
  string(REGEX REPLACE "[.]glsl$" ".h" _output_relative "${_source_relative}")
  set(_output_file "${CMAKE_CURRENT_BINARY_DIR}/${_output_relative}")
  set(_depfile "${_output_file}.d")

  string(REGEX REPLACE "[^A-Za-z0-9_]" "_" _logical_name "${_name}")
  string(TOUPPER "COIN_SHADER_${_logical_name}_H" COIN_HEADER_DEF)
  set(COIN_TEXTVAR_NAME "coin_${_logical_name}_shadersource")

  add_custom_command(
    OUTPUT "${_output_file}"
    BYPRODUCTS "${_depfile}"
    COMMAND "${CMAKE_COMMAND}"
      "-DINPUT_FILE=${_input_file}"
      "-DOUTPUT_FILE=${_output_file}"
      "-DDEPFILE=${_depfile}"
      "-DSOURCE_ROOT=${PROJECT_SOURCE_DIR}"
      "-DTEMPLATE_FILE=${CMAKE_CURRENT_SOURCE_DIR}/strfytemplate.cmake.in"
      "-DCOIN_HEADER_DEF=${COIN_HEADER_DEF}"
      "-DCOIN_TEXTVAR_NAME=${COIN_TEXTVAR_NAME}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/GenerateShaderHeader.cmake"
    DEPENDS
      "${_input_file}"
      "${CMAKE_CURRENT_SOURCE_DIR}/GenerateShaderHeader.cmake"
      "${CMAKE_CURRENT_SOURCE_DIR}/strfytemplate.cmake.in"
    DEPFILE "${_depfile}"
    VERBATIM
  )
  list(APPEND COIN_GENERATED_DATA_HEADERS "${_output_file}")
endmacro()

# GLSL roots are executable shader programs. Register them explicitly so
# included modules are only discovered by the shader generator and are never
# embedded as independent C++ resources.
coin_embed_gl_shader(gl_visual_vertex data/shaders/gl/visual/Vertex.glsl)
coin_embed_gl_shader(gl_visual_fragment data/shaders/gl/visual/Fragment.glsl)
coin_embed_gl_shader(gl_wide_line_vertex data/shaders/gl/wide-line/Vertex.glsl)
coin_embed_gl_shader(gl_wide_line_geometry data/shaders/gl/wide-line/Geometry.glsl)
coin_embed_gl_shader(gl_wide_line_fragment data/shaders/gl/wide-line/Fragment.glsl)
coin_embed_gl_shader(gl_point_vertex data/shaders/gl/point/Vertex.glsl)
coin_embed_gl_shader(gl_point_geometry data/shaders/gl/point/Geometry.glsl)
coin_embed_gl_shader(gl_point_fragment data/shaders/gl/point/Fragment.glsl)
coin_embed_gl_shader(gl_pixel_vertex data/shaders/gl/pixel/Vertex.glsl)
coin_embed_gl_shader(gl_pixel_fragment data/shaders/gl/pixel/Fragment.glsl)

# Vulkan SPIR-V headers are committed to the source tree
# (src/rendering/vulkan/visual/*.spv.h) so the backend builds without a
# runtime shader compiler.  Regeneration is an explicit, offline step driven
# by the `coin_regenerate_vulkan_spirv` target; it is intentionally not part
# of the default build so glslangValidator is never a build dependency.
#
# Regenerate with a single, pinned glslangValidator version and rebuild every
# header in one run.  A different toolchain version can emit equivalent-but-
# different SPIR-V, churning the vendored headers the next time they are
# regenerated; treat the committed headers as the reference.
if(COIN_BUILD_VULKAN_RENDERER)
  find_program(COIN_GLSLANG_VALIDATOR glslangValidator DOC
    "Path to glslangValidator for offline Vulkan SPIR-V regeneration")

  # Offline SPIR-V validation gate for the regeneration target.  Optional: the
  # committed .spv.h headers build without either tool, and a machine lacking
  # them simply skips validation (the step is a no-op, not a hard error).
  find_program(COIN_SPIRV_VAL spirv-val DOC
    "Path to spirv-val for offline Vulkan SPIR-V validation")
  find_program(COIN_SPIRV_OPT spirv-opt DOC
    "Path to spirv-opt for optional offline SPIR-V optimization checking")

  # Optional std140 layout reflection gate (spirv-dis based; the packaged
  # toolchains have no SPIRV-Reflect).  The superproject points these at
  # tools/rendering/spirv_layout_check.py + spirv_layouts.json before adding
  # this directory; a standalone Coin build leaves them empty and the step is
  # skipped.  It asserts the shader block layouts match the C++ mirrors
  # (VulkanPushConstants/VulkanBackgroundPush/VulkanLightingUbo/VulkanDrawUbo).
  find_program(COIN_SPIRV_DIS spirv-dis DOC
    "Path to spirv-dis for offline SPIR-V layout reflection")
  set(COIN_SPIRV_LAYOUT_CHECK "" CACHE FILEPATH
    "Path to spirv_layout_check.py (SPIR-V std140 layout reflection)")
  set(COIN_SPIRV_LAYOUTS "" CACHE FILEPATH
    "Path to spirv_layouts.json (expected std140 block layouts)")
  if(COIN_SPIRV_LAYOUT_CHECK)
    find_package(Python3 COMPONENTS Interpreter QUIET)
  endif()
  # Re-run the gate when the expected layouts change, not only when the shader
  # source changes.
  set(_coin_layout_deps "")
  foreach(_layout_file "${COIN_SPIRV_LAYOUT_CHECK}" "${COIN_SPIRV_LAYOUTS}")
    if(_layout_file AND EXISTS "${_layout_file}")
      list(APPEND _coin_layout_deps "${_layout_file}")
    endif()
  endforeach()

  # Shader-side diagnostics (debugPrintfEXT).  Off by default so the committed
  # .spv.h headers stay byte-identical; turning it on recompiles the shaders
  # with COIN_ENABLE_DEBUG_PRINTF and requires FC_VULKAN_DEBUG_PRINTF at run
  # time.  A shader opts in with the guarded pattern:
  #
  #   #ifdef COIN_ENABLE_DEBUG_PRINTF
  #   #extension GL_EXT_debug_printf : enable
  #   #endif
  #   ...
  #   #ifdef COIN_ENABLE_DEBUG_PRINTF
  #     debugPrintfEXT("value=%f", value);
  #   #endif
  #
  # (glslang 16.4 rejects variadic macros, so a COIN_DEBUG_PRINTF(...) helper
  # is deliberately not used.)
  option(COIN_VULKAN_DEBUG_PRINTF
    "Compile Vulkan shaders with COIN_ENABLE_DEBUG_PRINTF (VK_EXT_debug_printf)"
    OFF)
  set(_coin_glsl_define "")
  if(COIN_VULKAN_DEBUG_PRINTF)
    set(_coin_glsl_define "COIN_ENABLE_DEBUG_PRINTF")
  endif()

  set(COIN_VULKAN_SPIRV_REGENERATE_TARGETS "")

  macro(coin_embed_vulkan_spirv _name _input _stage)
    set(_input_file "${PROJECT_SOURCE_DIR}/${_input}")
    if(NOT EXISTS "${_input_file}")
      message(FATAL_ERROR "Vulkan GLSL shader does not exist: ${_input}")
    endif()

    get_filename_component(_basename "${_input}" NAME_WE)
    # Optional fourth argument: the output subdirectory under
    # src/rendering/vulkan/ (default "visual").  The RT shaders pass "rt" so
    # the generated header lands directly next to the backend that includes
    # it, instead of being written into visual/ and copied afterwards (which
    # left stray untracked copies there).
    set(_out_subdir "visual")
    if(${ARGC} GREATER 3)
      set(_out_subdir "${ARGV3}")
    endif()
    set(_output_file
      "${PROJECT_SOURCE_DIR}/src/rendering/vulkan/${_out_subdir}/${_basename}.spv.h")

    string(REGEX REPLACE "[^A-Za-z0-9_]" "_" _logical_name "${_name}")
    set(_variable_name "coin_${_logical_name}_spirv")
    set(_target_name "coin_regenerate_vulkan_spirv_${_logical_name}")

    # Ray tracing stages need SPIR-V 1.4+ (GL_EXT_ray_tracing); the visual
    # stages keep the vulkan1.0 default.
    set(_target_env vulkan1.0)
    if("${_stage}" STREQUAL "rgen" OR "${_stage}" STREQUAL "rmiss" OR
       "${_stage}" STREQUAL "rchit" OR "${_stage}" STREQUAL "rahit" OR
       "${_stage}" STREQUAL "rint")
      set(_target_env vulkan1.2)
    endif()

    add_custom_command(
      OUTPUT "${_output_file}"
      COMMAND "${CMAKE_COMMAND}"
        "-DGLSLANG_VALIDATOR=${COIN_GLSLANG_VALIDATOR}"
        "-DINPUT_FILE=${_input_file}"
        "-DOUTPUT_FILE=${_output_file}"
        "-DSTAGE=${_stage}"
        "-DVARIABLE_NAME=${_variable_name}"
        "-DTARGET_ENV=${_target_env}"
        "-DSPIRV_VAL=${COIN_SPIRV_VAL}"
        "-DSPIRV_OPT=${COIN_SPIRV_OPT}"
        "-DSPIRV_DIS=${COIN_SPIRV_DIS}"
        "-DSPIRV_LAYOUT_CHECK=${COIN_SPIRV_LAYOUT_CHECK}"
        "-DSPIRV_LAYOUTS=${COIN_SPIRV_LAYOUTS}"
        "-DPYTHON_EXECUTABLE=${Python3_EXECUTABLE}"
        "-DGLSL_DEFINE=${_coin_glsl_define}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/GenerateVulkanSpirvHeader.cmake"
      DEPENDS
        "${_input_file}"
        "${CMAKE_CURRENT_SOURCE_DIR}/GenerateVulkanSpirvHeader.cmake"
        ${_coin_layout_deps}
      VERBATIM
      COMMENT "Regenerating Vulkan SPIR-V for ${_name} (${_stage})"
    )
    add_custom_target(${_target_name}
      DEPENDS "${_output_file}"
    )
    list(APPEND COIN_VULKAN_SPIRV_REGENERATE_TARGETS "${_target_name}")
  endmacro()

  if(NOT COIN_GLSLANG_VALIDATOR)
    message(WARNING
      "glslangValidator not found; the coin_regenerate_vulkan_spirv target is unavailable")
  else()
    if(NOT COIN_SPIRV_VAL)
      message(STATUS
        "spirv-val not found; SPIR-V regeneration will skip validation")
    else()
      message(STATUS
        "SPIR-V regeneration will validate with ${COIN_SPIRV_VAL}")
    endif()
    coin_embed_vulkan_spirv(vulkan_visual_vertex
      data/shaders/vulkan/visual/Vertex.glsl vert)
    coin_embed_vulkan_spirv(vulkan_visual_fragment
      data/shaders/vulkan/visual/Fragment.glsl frag)
    coin_embed_vulkan_spirv(vulkan_wide_line_vertex
      data/shaders/vulkan/wide-line/WideLineVertex.glsl vert)
    coin_embed_vulkan_spirv(vulkan_wide_line_instanced_vertex
      data/shaders/vulkan/wide-line/WideLineInstancedVertex.glsl vert)
    coin_embed_vulkan_spirv(vulkan_wide_line_fragment
      data/shaders/vulkan/wide-line/WideLineFragment.glsl frag)
    coin_embed_vulkan_spirv(vulkan_geometry_lod_subpixel_cull
      data/shaders/vulkan/geometry-lod/SubPixelCull.glsl comp)
    coin_embed_vulkan_spirv(vulkan_background_vertex
      data/shaders/vulkan/visual/BackgroundVertex.glsl vert)
    coin_embed_vulkan_spirv(vulkan_background_fragment
      data/shaders/vulkan/visual/BackgroundFragment.glsl frag)

    # Ray-tracing backend shaders.  These pass the "rt" output subdirectory so
    # the generated headers land in src/rendering/vulkan/rt/ where the backend
    # includes them.  The .spv.h files are committed like the visual ones; this
    # is only the offline regeneration path.
    coin_embed_vulkan_spirv(vulkan_rt_pathtrace
      data/shaders/vulkan/rt/PathTrace.glsl comp rt)
    coin_embed_vulkan_spirv(vulkan_rt_raygen
      data/shaders/vulkan/rt/Raygen.glsl rgen rt)
    coin_embed_vulkan_spirv(vulkan_rt_closesthit
      data/shaders/vulkan/rt/ClosestHit.glsl rchit rt)
    coin_embed_vulkan_spirv(vulkan_rt_miss
      data/shaders/vulkan/rt/Miss.glsl rmiss rt)
    coin_embed_vulkan_spirv(vulkan_rt_shadowmiss
      data/shaders/vulkan/rt/ShadowMiss.glsl rmiss rt)
    coin_embed_vulkan_spirv(vulkan_rt_shadowclosesthit
      data/shaders/vulkan/rt/ShadowClosestHit.glsl rchit rt)
    coin_embed_vulkan_spirv(vulkan_rt_presentvertex
      data/shaders/vulkan/rt/PresentVertex.glsl vert rt)
    coin_embed_vulkan_spirv(vulkan_rt_presentfragment
      data/shaders/vulkan/rt/PresentFragment.glsl frag rt)
    coin_embed_vulkan_spirv(vulkan_rt_denoisedownsample
      data/shaders/vulkan/rt/DenoiseDownsample.glsl comp rt)
    # Single-ray GPU pick for the RTX viewport (Vulkan/RTX only).
    coin_embed_vulkan_spirv(vulkan_rt_pick
      data/shaders/vulkan/rt/Pick.glsl comp rt)

    add_custom_target(coin_regenerate_vulkan_spirv
      DEPENDS ${COIN_VULKAN_SPIRV_REGENERATE_TARGETS}
      COMMENT "Regenerating all committed Vulkan SPIR-V headers")
  endif()
endif()

# These are legacy shader snippets exposed through Coin's built-in shader
# dictionary. They are ordinary embedded text, not core-profile root
# programs, and therefore remain on the compatibility resource path.
set(COIN_LEGACY_SHADER_RESOURCES
  data/shaders/lights/DirSpotLight.glsl
  data/shaders/lights/DirectionalLight.glsl
  data/shaders/lights/PointLight.glsl
  data/shaders/lights/SpotLight.glsl
  data/shaders/vsm/VsmLookup.glsl
)
foreach(_input_file ${COIN_LEGACY_SHADER_RESOURCES})
  coin_embed_text("${_input_file}" "_shadersource")
endforeach()

# Inventor resources remain ordinary embedded text. This path deliberately
# has no shader parsing or include semantics.
file(GLOB_RECURSE COIN_SHADER_INVENTOR RELATIVE "${PROJECT_SOURCE_DIR}"
  "${PROJECT_SOURCE_DIR}/*.iv")
foreach(_input_file ${COIN_SHADER_INVENTOR})
  coin_embed_text("${_input_file}" "_draggergeometry")
endforeach()

# Keep generated text and shader headers on the build graph. The object
# libraries that include them depend on this target below.
add_custom_target(coin_generated_data_headers
  DEPENDS ${COIN_GENERATED_DATA_HEADERS}
)

file(GLOB_RECURSE COMMON_SCXML "${CMAKE_CURRENT_SOURCE_DIR}/[cep]*.xml")
file(GLOB_RECURSE COMMON_DRAGGERDEFAULTS "${CMAKE_CURRENT_SOURCE_DIR}/*.iv")
file(GLOB_RECURSE COMMON_SHADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.glsl")

set(COMMON_RESOURCES
  ${COMMON_SCXML}
  ${COMMON_DRAGGERDEFAULTS}
  ${COMMON_SHADERS}
  PARENT_SCOPE
)
