# CoinCompileSpirv.cmake
#
# Build-time compilation of Coin's Vulkan GLSL sources to SPIR-V, embedded as
# generated C headers.  The .glsl files under data/shaders/vulkan are the single
# source of truth; no generated shader data is checked in.
#
# Usage (from src/rendering/CMakeLists.txt):
#   include(cmake/CoinCompileSpirv.cmake)
#   coin_compile_vulkan_shaders(<out-var>)
#
# Requires glslangValidator (conda-forge 'glslang', Debian/Ubuntu
# 'glslang-tools', Windows vcpkg/conda 'glslang').

find_program(COIN_GLSLANG_VALIDATOR
  NAMES glslangValidator glslang
  DOC "glslangValidator, used to compile the Vulkan renderer's GLSL to SPIR-V")

function(coin_compile_vulkan_shaders _out_var)
  if(NOT COIN_GLSLANG_VALIDATOR)
    message(FATAL_ERROR
      "COIN_BUILD_VULKAN_RENDERER=ON requires glslangValidator, but it was not "
      "found.  Install the 'glslang' package (conda-forge / vcpkg) or "
      "'glslang-tools' (Debian/Ubuntu), or set COIN_GLSLANG_VALIDATOR.")
  endif()

  # <source path relative to data/shaders/vulkan>|<C symbol>|<shader stage>
  set(_specs
    "visual/Fragment|coin_vulkan_visual_fragment_spirv|frag"
    "visual/Vertex|coin_vulkan_visual_vertex_spirv|vert"
    "visual/BackgroundFragment|coin_vulkan_background_fragment_spirv|frag"
    "visual/BackgroundVertex|coin_vulkan_background_vertex_spirv|vert"
    "wide-line/WideLineFragment|coin_vulkan_wide_line_fragment_spirv|frag"
    "wide-line/WideLineVertex|coin_vulkan_wide_line_vertex_spirv|vert"
    "wide-line/WideLineInstancedVertex|coin_vulkan_wide_line_instanced_vertex_spirv|vert"
    "geometry-lod/SubPixelCull|coin_vulkan_geometry_lod_subpixel_cull_spirv|comp"
  )

  set(_headers "")
  foreach(_spec ${_specs})
    string(REPLACE "|" ";" _parts "${_spec}")
    list(GET _parts 0 _rel)
    list(GET _parts 1 _var)
    list(GET _parts 2 _stage)

    set(_source "${PROJECT_SOURCE_DIR}/data/shaders/vulkan/${_rel}.glsl")
    # All generated headers live in vulkan/visual/ regardless of the source
    # subdirectory, matching the include paths used by the Vulkan backend.
    get_filename_component(_name "${_rel}" NAME)
    set(_header "${CMAKE_CURRENT_BINARY_DIR}/vulkan/visual/${_name}.spv.h")

    if(NOT EXISTS "${_source}")
      message(FATAL_ERROR "Vulkan shader source not found: ${_source}")
    endif()

    add_custom_command(
      OUTPUT "${_header}"
      COMMAND "${CMAKE_COMMAND}"
              "-DCOIN_GLSLANG_VALIDATOR=${COIN_GLSLANG_VALIDATOR}"
              "-DCOIN_SHADER_SOURCE=${_source}"
              "-DCOIN_SHADER_STAGE=${_stage}"
              "-DCOIN_SHADER_HEADER=${_header}"
              "-DCOIN_SHADER_VARIABLE=${_var}"
              "-DCOIN_SHADER_SOURCE_REL=data/shaders/vulkan/${_rel}.glsl"
              -P "${PROJECT_SOURCE_DIR}/cmake/CoinCompileSpirvEmbed.cmake"
      DEPENDS "${_source}"
      COMMENT "Compiling Vulkan shader ${_rel}.glsl to SPIR-V"
      VERBATIM)

    list(APPEND _headers "${_header}")
  endforeach()

  set(${_out_var} "${_headers}" PARENT_SCOPE)
endfunction()
