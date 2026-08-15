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
if(COIN_BUILD_VULKAN_RENDERER)
  find_program(COIN_GLSLANG_VALIDATOR glslangValidator DOC
    "Path to glslangValidator for offline Vulkan SPIR-V regeneration")

  set(COIN_VULKAN_SPIRV_REGENERATE_TARGETS "")

  macro(coin_embed_vulkan_spirv _name _input _stage)
    set(_input_file "${PROJECT_SOURCE_DIR}/${_input}")
    if(NOT EXISTS "${_input_file}")
      message(FATAL_ERROR "Vulkan GLSL shader does not exist: ${_input}")
    endif()

    get_filename_component(_basename "${_input}" NAME_WE)
    set(_output_file "${PROJECT_SOURCE_DIR}/src/rendering/vulkan/visual/${_basename}.spv.h")

    string(REGEX REPLACE "[^A-Za-z0-9_]" "_" _logical_name "${_name}")
    set(_variable_name "coin_${_logical_name}_spirv")
    set(_target_name "coin_regenerate_vulkan_spirv_${_logical_name}")

    add_custom_command(
      OUTPUT "${_output_file}"
      COMMAND "${CMAKE_COMMAND}"
        "-DGLSLANG_VALIDATOR=${COIN_GLSLANG_VALIDATOR}"
        "-DINPUT_FILE=${_input_file}"
        "-DOUTPUT_FILE=${_output_file}"
        "-DSTAGE=${_stage}"
        "-DVARIABLE_NAME=${_variable_name}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/GenerateVulkanSpirvHeader.cmake"
      DEPENDS
        "${_input_file}"
        "${CMAKE_CURRENT_SOURCE_DIR}/GenerateVulkanSpirvHeader.cmake"
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
    coin_embed_vulkan_spirv(vulkan_visual_vertex
      data/shaders/vulkan/visual/Vertex.glsl vert)
    coin_embed_vulkan_spirv(vulkan_visual_fragment
      data/shaders/vulkan/visual/Fragment.glsl frag)

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
