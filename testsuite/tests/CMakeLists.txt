# ============================================================================
# Coin fork test suite.
#
# The standalone tests below are grouped by subsystem, mirroring the layout of
# the Coin sources under src/.  Every test is registered through
# coin_add_test(), which centralizes the executable/link/include/CTest wiring,
# so adding a test is a single line (or a single table entry).
#
#   coin_add_test(<target> <source relative to tests/> [options...])
#
# Options:
#   CXX20               build the target with C++20 (required by Vulkan tests)
#   SKIP77              treat exit code 77 as "skipped" (no usable GPU/device)
#   SOURCES <...>       extra sources compiled into the target
#   INCLUDE_DIRS <...>  extra include directories
#   DEFINES <...>       extra compile definitions
#   DEPENDS <...>       extra build dependencies
#   LABELS <...>        CTest labels
# ============================================================================

set(COIN_TEST_SKIP_RETURN_CODE 77)

function(coin_add_test target source)
	cmake_parse_arguments(ARG
		"CXX20;SKIP77"
		""
		"SOURCES;INCLUDE_DIRS;DEFINES;DEPENDS;LABELS"
		${ARGN})

	add_executable(${target} "${CMAKE_CURRENT_SOURCE_DIR}/${source}" ${ARG_SOURCES})
	target_link_libraries(${target} Coin ${COIN_TARGET_LINK_LIBRARIES})
	target_include_directories(${target} PRIVATE
		${PROJECT_SOURCE_DIR}/src
		${PROJECT_SOURCE_DIR}/include
		${PROJECT_BINARY_DIR}/include
		${PROJECT_SOURCE_DIR}/testsuite
		${ARG_INCLUDE_DIRS}
		${COIN_TARGET_INCLUDE_DIRECTORIES}
	)
	if(ARG_DEFINES)
		target_compile_definitions(${target} PRIVATE ${ARG_DEFINES})
	endif()
	if(ARG_CXX20)
		set_target_properties(${target} PROPERTIES
			CXX_STANDARD 20
			CXX_STANDARD_REQUIRED ON
		)
	endif()
	if(ARG_DEPENDS)
		add_dependencies(${target} ${ARG_DEPENDS})
	endif()

	add_test(NAME ${target} COMMAND ${target})

	set(_properties "")
	if(ARG_SKIP77)
		list(APPEND _properties SKIP_RETURN_CODE ${COIN_TEST_SKIP_RETURN_CODE})
	endif()
	if(ARG_LABELS)
		list(APPEND _properties LABELS ${ARG_LABELS})
	endif()
	if(_properties)
		set_tests_properties(${target} PROPERTIES ${_properties})
	endif()
endfunction()

# ---------------------------------------------------------------------------
# core/ - scene-graph traversal and render-manager core
# ---------------------------------------------------------------------------
coin_add_test(CoreTraversalTest core/core-traversal-test.cpp)
if(NOT COIN_BUILD_LEGACY_GL_RENDERER)
	coin_add_test(SceneManagerCoreTest core/scene-manager-core-test.cpp)
endif()

# ---------------------------------------------------------------------------
# retained/ - retained renderer IR, nodes, materials, raster text and drawlist
# ---------------------------------------------------------------------------
coin_add_test(RetainedIRTest retained/retained-ir-test.cpp)
coin_add_test(RetainedNodeTest retained/retained-node-test.cpp)
coin_add_test(RetainedMixedTopologyTest retained/retained-mixed-topology-test.cpp)
coin_add_test(RetainedMaterialLightingTest retained/retained-material-lighting-test.cpp)
coin_add_test(RetainedMaterialCoalesceTest retained/retained-material-coalesce-test.cpp)
coin_add_test(RetainedRasterTextTest retained/retained-raster-text-test.cpp)

# ---------------------------------------------------------------------------
# model/ - pure-CPU host contract tests (no GPU or window system)
# ---------------------------------------------------------------------------
coin_add_test(PhysicalMaterialElementTest model/physical-material-element-test.cpp)
coin_add_test(MaterialBlockPackTest model/material-block-pack-test.cpp)
coin_add_test(TextureCoordinateProjectionTest model/texture-coordinate-projection-test.cpp)
coin_add_test(NodeSerializationTest model/node-serialization-test.cpp)

# ---------------------------------------------------------------------------
# gl/ - legacy OpenGL backend, GLSL, EGL and offscreen readback
# ---------------------------------------------------------------------------
coin_add_test(LegacyBlendingTest gl/legacy-blending-test.cpp)

if(HAVE_EGL)
	coin_add_test(RetainedMaterialLightingGLTest retained/retained-material-lighting-gl-test.cpp
		SKIP77
		DEFINES COIN_INTERNAL
		SOURCES ${PROJECT_SOURCE_DIR}/src/rendering/gl/SoGLRenderBackend.cpp
		INCLUDE_DIRS ${PROJECT_BINARY_DIR}/data
		DEPENDS coin_generated_data_headers)
	coin_add_test(RetainedRasterGLTest retained/retained-raster-gl-test.cpp
		SKIP77
		DEFINES COIN_INTERNAL
		SOURCES ${PROJECT_SOURCE_DIR}/src/rendering/gl/SoGLRenderBackend.cpp
		INCLUDE_DIRS ${PROJECT_BINARY_DIR}/data
		DEPENDS coin_generated_data_headers)
	coin_add_test(DrawListGLTest gl/drawlist-gl-test.cpp
		SKIP77
		DEFINES COIN_INTERNAL
		SOURCES ${PROJECT_SOURCE_DIR}/src/rendering/gl/SoGLRenderBackend.cpp
		INCLUDE_DIRS ${PROJECT_BINARY_DIR}/data
		DEPENDS coin_generated_data_headers)
	coin_add_test(DrawListSortStabilityTest retained/drawlist-sort-stability-test.cpp
		SKIP77
		DEFINES COIN_INTERNAL)
	coin_add_test(EGLBindingTest gl/egl-binding-test.cpp
		SKIP77
		DEFINES COIN_INTERNAL)
	coin_add_test(GLSLRuntimeTest gl/glsl-runtime-test.cpp
		SKIP77
		DEFINES COIN_INTERNAL)
	coin_add_test(GLSLDiagnosticsTest gl/glsl-diagnostics-test.cpp
		SKIP77
		DEFINES COIN_INTERNAL)
	coin_add_test(OffscreenReadbackTest gl/offscreen-readback-test.cpp
		SKIP77
		DEFINES COIN_INTERNAL)

	# -----------------------------------------------------------------------
	# vulkan/ - headless Vulkan backend tests (need a usable Vulkan device)
	# -----------------------------------------------------------------------
	if(COIN_BUILD_VULKAN_RENDERER)
		# "target source" pairs; every test skips with code 77 when no
		# usable Vulkan device is present.
		set(COIN_VULKAN_TESTS
			DrawListVulkanTest vulkan/drawlist-vulkan-test.cpp
			VulkanBackendLifecycleTest vulkan/vulkan-backend-lifecycle-test.cpp
			VulkanBackendRenderExternalTest vulkan/vulkan-backend-render-external-test.cpp
			VulkanBackendTextOverlayTest vulkan/vulkan-backend-text-overlay-test.cpp
			VulkanBackendMixedTopologyTest vulkan/vulkan-backend-mixed-topology-test.cpp
			VulkanBackendFragmentOutputTest vulkan/vulkan-backend-fragment-output-test.cpp
			VulkanBackendDepthTest vulkan/vulkan-backend-depth-test.cpp
			VulkanBackendCullingTest vulkan/vulkan-backend-culling-test.cpp
			VulkanBackendLightingTest vulkan/vulkan-backend-lighting-test.cpp
			VulkanBackendTextureModelsTest vulkan/vulkan-backend-texture-models-test.cpp
			VulkanBackendMultitexStagingTest vulkan/vulkan-backend-multitex-staging-test.cpp
			VulkanBackendScissorTest vulkan/vulkan-backend-scissor-test.cpp
			VulkanBackendClearTest vulkan/vulkan-backend-clear-test.cpp
			VulkanBackendYFlipTest vulkan/vulkan-backend-yflip-test.cpp
			VulkanBackendStencilTest vulkan/vulkan-backend-stencil-test.cpp
			VulkanBackendTessOverlayTest vulkan/vulkan-backend-tess-overlay-test.cpp
			VulkanBackendMaterialTest vulkan/vulkan-backend-material-test.cpp
			VulkanBackendHdrTest vulkan/vulkan-backend-hdr-test.cpp
			VulkanBackendRetainedChangeTest vulkan/vulkan-backend-retained-change-test.cpp
			VulkanBackendMaterialChangeTest vulkan/vulkan-backend-material-change-test.cpp
			VulkanBackendParallelTest vulkan/vulkan-backend-parallel-test.cpp
			VulkanRenderManagerTest vulkan/vulkan-rendermanager-test.cpp
			VulkanRenderManagerVisibilityTest vulkan/vulkan-rendermanager-visibility-test.cpp
			VulkanLightStabilityTest vulkan/vulkan-light-stability-test.cpp
		)
		list(LENGTH COIN_VULKAN_TESTS _count)
		math(EXPR _pairs "${_count} / 2")
		set(_index 0)
		while(_index LESS _pairs)
			math(EXPR _offset "${_index} * 2")
			list(GET COIN_VULKAN_TESTS ${_offset} _name)
			math(EXPR _source_offset "${_offset} + 1")
			list(GET COIN_VULKAN_TESTS ${_source_offset} _source)
			coin_add_test(${_name} ${_source} CXX20 SKIP77 LABELS vulkan)
			math(EXPR _index "${_index} + 1")
		endwhile()
	endif()
endif()

# ---------------------------------------------------------------------------
# scripts/ - CMake-driven generator checks
# ---------------------------------------------------------------------------
add_test(
	NAME ShaderGeneratorTest
	COMMAND "${CMAKE_COMMAND}"
		-DCOIN_SOURCE_DIR=${PROJECT_SOURCE_DIR}
		-DCOIN_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
		"-DCOIN_GENERATOR=${CMAKE_GENERATOR}"
		-P "${CMAKE_CURRENT_SOURCE_DIR}/scripts/shader-generator-test.cmake"
)
