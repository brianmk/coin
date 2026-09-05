macro(create_testsuite input)
	get_filename_component(FLNAME "${input}" NAME_WE)
	get_filename_component(FLPATH "${input}" PATH)
	string(REGEX REPLACE ".*[/\\]" "" FLSUBFLD "${FLPATH}")
	set(COIN_STR_TEST_CLASS "${FLNAME}")
	file(READ ${PROJECT_SOURCE_DIR}/${input} f0)
	if(f0 MATCHES "#ifdef[ \t]+COIN_TEST_SUITE")
		# message(STATUS "Parse: ${PROJECT_SOURCE_DIR}/${input} - ${FLPATHSUB}${FLNAME}Test.cpp")
		# get first include from file, which we assume is include to tested class
		string(REGEX MATCH "[\n\r]+#include[ \t]<[^\n]+" iclass "${f0}")
		# get block between '#ifdef COIN_TEST_SUITE' and '#endif'
		string(REGEX REPLACE ".*#ifdef[ \t]+COIN_TEST_SUITE" "" f1 "${f0}")
		string(REGEX REPLACE "#endif[ \t/!]+COIN_TEST_SUITE.*" "" f2 "${f1}")
		# get all #include statements within COIN_TEST_SUITE code block
		string(REGEX MATCHALL "#include[ \t]<[^\n]+" i0 "${f2}")
		string(REPLACE ";" "\n" COIN_STR_TEST_INCL "${i0}")
		set(COIN_STR_TEST_INCL "${iclass}\n${COIN_STR_TEST_INCL}")
		# remove #include statements from test code string (moved to ${COIN_STR_TEST_INCL})
		string(REGEX REPLACE "[\n\r ]*#include[ \t]<[^\n]+" "" COIN_STR_TEST_CODE "${f2}")
		# generate new test code file with extracted snippets
		configure_file(TestSuiteTemplate.cmake.in "${FLSUBFLD}${FLNAME}Test.cpp")
	endif()
endmacro()

macro(create_file_test input testfunc filter)
	file(GLOB_RECURSE COIN_IV_FILES RELATIVE ${PROJECT_SOURCE_DIR}
		../${input}/*.iv
		../${input}/*.iv.gz
	)
	if (HAVE_VRML97)
		file(GLOB_RECURSE COIN_WRL_VRML_FILES RELATIVE ${PROJECT_SOURCE_DIR}
			../${input}/*.wrl
			../${input}/*.vrml
			../${input}/*.wrl.gz
			../${input}/*.vrml.gz
		)
		list(APPEND COIN_IV_FILES ${COIN_WRL_VRML_FILES})
	endif()
	string(REGEX REPLACE "[\\./\\-]+" "" TESTSUITENAME "${input}")
	set(COIN_STR_TEST_INCL "")
	set(COIN_STR_TEST_CODE "const char *readError[] = { \"Coin read error:\", 0 };\n")
	set(COIN_STR_TEST_CLASS "${TESTSUITENAME}")
	set(extra_macro_args ${ARGN})
	list(LENGTH extra_macro_args num_extra_args)
	foreach(INPUTFILE ${COIN_IV_FILES})
		if(${num_extra_args} EQUAL 0 OR NOT ("${INPUTFILE}" MATCHES "${extra_macro_args}"))
			string(REGEX REPLACE "[\\./\\-]+" "_" TESTNAME "${INPUTFILE}")
			if (${filter})
				set(COIN_STR_TEST_CODE "${COIN_STR_TEST_CODE}\nBOOST_AUTO_TEST_CASE(${TESTNAME}) { PushMessageSuppressFilters(readError); ResetReadErrorCount(); test_file(\"${PROJECT_SOURCE_DIR}/${INPUTFILE}\", &${testfunc}); PopMessageSuppressFilters(); }")
			else()
				set(COIN_STR_TEST_CODE "${COIN_STR_TEST_CODE}\nBOOST_AUTO_TEST_CASE(${TESTNAME}) { ResetReadErrorCount(); test_file(\"${PROJECT_SOURCE_DIR}/${INPUTFILE}\", &${testfunc}); }")
			endif()
		endif()
	endforeach()
	configure_file(TestSuiteTemplate.cmake.in "${TESTSUITENAME}Test.cpp")
endmacro()

# Parse all source files for embedded '#ifdef COIN_TEST_SUITE' and extract those blocks
# to separate test source files.
file(GLOB_RECURSE COIN_SRC_FILES RELATIVE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src/*.cpp)
foreach(INPUTFILE ${COIN_SRC_FILES})
	create_testsuite(${INPUTFILE})
endforeach()

# Create test cases for all scenes under models folder (this replaces StandardTests.cpp).
# The original StandardTests.cpp also tested additional folders:
# - "killers" with testInCorrectFile
# - "slackers" with testOutOfSpecFile
# However, those folders are not in the repository.
if(COIN_BUILD_LEGACY_GL_RENDERER)
	create_file_test("models" "testCorrectFile" false "incorrect")
else()
	# ExtSelection is part of the legacy renderer and is not registered in
	# core-only builds, so do not include scenes that require that node.
	create_file_test("models" "testCorrectFile" false "(incorrect|models/dead_simple/all-iv\\.iv$)")
endif()
# bug039.iv and bug055-5.iv contain errors but load anyway: ignore for now
create_file_test("models/io/incorrect" "testIncorrectFile" true "(bug039|bug055-5)")

# Include all extracted '*Tests.cpp' files in the target.
FILE(GLOB COIN_TEST_SOURCES "${CMAKE_CURRENT_BINARY_DIR}/*Test.cpp")
if(NOT COIN_BUILD_LEGACY_GL_RENDERER)
  list(FILTER COIN_TEST_SOURCES EXCLUDE REGEX "/shadowsSo.*Test\\.cpp$")
endif()

add_executable(CoinTests TestSuiteMain.cpp TestSuiteUtils.cpp TestSuiteMisc.cpp ${COIN_TEST_SOURCES})
set_target_properties(CoinTests PROPERTIES DEBUG_POSTFIX "${CMAKE_DEBUG_POSTFIX}")
target_link_libraries(CoinTests Coin ${COIN_TARGET_LINK_LIBRARIES})
target_include_directories(CoinTests PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}
	${PROJECT_SOURCE_DIR}/include
	${PROJECT_SOURCE_DIR}/include/Inventor/annex
	${PROJECT_BINARY_DIR}/include
	${COIN_TARGET_INCLUDE_DIRECTORIES}
)
if (USE_PTHREAD)
	target_link_libraries(CoinTests pthread)
endif()
add_test(NAME CoinTests COMMAND CoinTests)

add_test(
	NAME ShaderGeneratorTest
	COMMAND "${CMAKE_COMMAND}"
		-DCOIN_SOURCE_DIR=${PROJECT_SOURCE_DIR}
		-DCOIN_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
		"-DCOIN_GENERATOR=${CMAKE_GENERATOR}"
		-P "${CMAKE_CURRENT_SOURCE_DIR}/shader-generator-test.cmake"
)

add_executable(CoreTraversalTest core-traversal-test.cpp)
target_link_libraries(CoreTraversalTest Coin ${COIN_TARGET_LINK_LIBRARIES})
target_include_directories(CoreTraversalTest PRIVATE
	${PROJECT_SOURCE_DIR}/include
	${PROJECT_BINARY_DIR}/include
	${COIN_TARGET_INCLUDE_DIRECTORIES}
)
add_test(NAME CoreTraversalTest COMMAND CoreTraversalTest)

if(NOT COIN_BUILD_LEGACY_GL_RENDERER)
	add_executable(SceneManagerCoreTest scene-manager-core-test.cpp)
	target_link_libraries(SceneManagerCoreTest Coin ${COIN_TARGET_LINK_LIBRARIES})
	target_include_directories(SceneManagerCoreTest PRIVATE
		${PROJECT_SOURCE_DIR}/include
		${PROJECT_BINARY_DIR}/include
		${COIN_TARGET_INCLUDE_DIRECTORIES}
	)
	add_test(NAME SceneManagerCoreTest COMMAND SceneManagerCoreTest)
endif()

add_executable(LegacyBlendingTest legacy-blending-test.cpp)
target_link_libraries(LegacyBlendingTest Coin ${COIN_TARGET_LINK_LIBRARIES})
target_include_directories(LegacyBlendingTest PRIVATE
	${PROJECT_SOURCE_DIR}/include
	${PROJECT_BINARY_DIR}/include
	${COIN_TARGET_INCLUDE_DIRECTORIES}
)
add_test(NAME LegacyBlendingTest COMMAND LegacyBlendingTest)

add_executable(RetainedIRTest retained-ir-test.cpp)
target_link_libraries(RetainedIRTest Coin ${COIN_TARGET_LINK_LIBRARIES})
target_include_directories(RetainedIRTest PRIVATE
	${PROJECT_SOURCE_DIR}/include
	${PROJECT_BINARY_DIR}/include
	${COIN_TARGET_INCLUDE_DIRECTORIES}
)
add_test(NAME RetainedIRTest COMMAND RetainedIRTest)

add_executable(RetainedNodeTest retained-node-test.cpp)
target_link_libraries(RetainedNodeTest Coin ${COIN_TARGET_LINK_LIBRARIES})
target_include_directories(RetainedNodeTest PRIVATE
	${PROJECT_SOURCE_DIR}/include
	${PROJECT_BINARY_DIR}/include
	${COIN_TARGET_INCLUDE_DIRECTORIES}
)
add_test(NAME RetainedNodeTest COMMAND RetainedNodeTest)

add_executable(RetainedMixedTopologyTest retained-mixed-topology-test.cpp)
target_link_libraries(RetainedMixedTopologyTest Coin ${COIN_TARGET_LINK_LIBRARIES})
target_include_directories(RetainedMixedTopologyTest PRIVATE
	${PROJECT_SOURCE_DIR}/include
	${PROJECT_BINARY_DIR}/include
	${COIN_TARGET_INCLUDE_DIRECTORIES}
)
add_executable(RetainedMaterialLightingTest retained-material-lighting-test.cpp)
target_link_libraries(RetainedMaterialLightingTest Coin ${COIN_TARGET_LINK_LIBRARIES})
target_include_directories(RetainedMaterialLightingTest PRIVATE
	${PROJECT_SOURCE_DIR}/include
	${PROJECT_BINARY_DIR}/include
	${COIN_TARGET_INCLUDE_DIRECTORIES}
)
add_test(NAME RetainedMixedTopologyTest COMMAND RetainedMixedTopologyTest)

add_test(NAME RetainedMaterialLightingTest COMMAND RetainedMaterialLightingTest)

add_executable(RetainedRasterTextTest retained-raster-text-test.cpp)
target_link_libraries(RetainedRasterTextTest Coin ${COIN_TARGET_LINK_LIBRARIES})
target_include_directories(RetainedRasterTextTest PRIVATE
	${PROJECT_SOURCE_DIR}/include
	${PROJECT_BINARY_DIR}/include
	${COIN_TARGET_INCLUDE_DIRECTORIES}
)
add_test(NAME RetainedRasterTextTest COMMAND RetainedRasterTextTest)
if(HAVE_EGL)
	add_executable(RetainedMaterialLightingGLTest retained-material-lighting-gl-test.cpp)
	target_link_libraries(RetainedMaterialLightingGLTest Coin ${COIN_TARGET_LINK_LIBRARIES})
	target_compile_definitions(RetainedMaterialLightingGLTest PRIVATE COIN_INTERNAL)
	target_include_directories(RetainedMaterialLightingGLTest PRIVATE
		${PROJECT_SOURCE_DIR}/src
		${PROJECT_SOURCE_DIR}/include
		${PROJECT_BINARY_DIR}/include
		${COIN_TARGET_INCLUDE_DIRECTORIES}
	)
	add_test(NAME RetainedMaterialLightingGLTest COMMAND RetainedMaterialLightingGLTest)
	set_tests_properties(RetainedMaterialLightingGLTest PROPERTIES SKIP_RETURN_CODE 77)

	add_executable(RetainedRasterGLTest retained-raster-gl-test.cpp)
	target_link_libraries(RetainedRasterGLTest Coin ${COIN_TARGET_LINK_LIBRARIES})
	target_compile_definitions(RetainedRasterGLTest PRIVATE COIN_INTERNAL)
	target_include_directories(RetainedRasterGLTest PRIVATE
		${PROJECT_SOURCE_DIR}/src
		${PROJECT_SOURCE_DIR}/include
		${PROJECT_BINARY_DIR}/include
		${COIN_TARGET_INCLUDE_DIRECTORIES}
	)
	add_test(NAME RetainedRasterGLTest COMMAND RetainedRasterGLTest)
	set_tests_properties(RetainedRasterGLTest PROPERTIES SKIP_RETURN_CODE 77)

	add_executable(DrawListGLTest drawlist-gl-test.cpp)
	target_link_libraries(DrawListGLTest Coin ${COIN_TARGET_LINK_LIBRARIES})
	target_compile_definitions(DrawListGLTest PRIVATE COIN_INTERNAL)
	target_include_directories(DrawListGLTest PRIVATE
		${PROJECT_SOURCE_DIR}/src
		${PROJECT_SOURCE_DIR}/include
		${PROJECT_BINARY_DIR}/include
		${COIN_TARGET_INCLUDE_DIRECTORIES}
	)
	add_test(NAME DrawListGLTest COMMAND DrawListGLTest)
	set_tests_properties(DrawListGLTest PROPERTIES SKIP_RETURN_CODE 77)

	add_executable(DrawListSortStabilityTest drawlist-sort-stability-test.cpp)
	target_link_libraries(DrawListSortStabilityTest Coin ${COIN_TARGET_LINK_LIBRARIES})
	target_compile_definitions(DrawListSortStabilityTest PRIVATE COIN_INTERNAL)
	target_include_directories(DrawListSortStabilityTest PRIVATE
		${PROJECT_SOURCE_DIR}/src
		${PROJECT_SOURCE_DIR}/include
		${PROJECT_BINARY_DIR}/include
		${COIN_TARGET_INCLUDE_DIRECTORIES}
	)
	add_test(NAME DrawListSortStabilityTest COMMAND DrawListSortStabilityTest)

	if(COIN_BUILD_VULKAN_RENDERER)
		add_executable(DrawListVulkanTest vulkan/drawlist-vulkan-test.cpp)
		target_link_libraries(DrawListVulkanTest Coin ${COIN_TARGET_LINK_LIBRARIES})
		target_compile_definitions(DrawListVulkanTest PRIVATE COIN_INTERNAL)
		target_include_directories(DrawListVulkanTest PRIVATE
			${PROJECT_SOURCE_DIR}/src
			${PROJECT_SOURCE_DIR}/include
			${PROJECT_BINARY_DIR}/include
			${PROJECT_SOURCE_DIR}/testsuite
			${COIN_TARGET_INCLUDE_DIRECTORIES}
		)
		add_test(NAME DrawListVulkanTest COMMAND DrawListVulkanTest)
		set_tests_properties(DrawListVulkanTest PROPERTIES SKIP_RETURN_CODE 77)

		add_executable(VulkanBackendLifecycleTest vulkan/vulkan-backend-lifecycle-test.cpp)
		target_link_libraries(VulkanBackendLifecycleTest Coin ${COIN_TARGET_LINK_LIBRARIES})
		target_compile_definitions(VulkanBackendLifecycleTest PRIVATE COIN_INTERNAL)
		target_include_directories(VulkanBackendLifecycleTest PRIVATE
			${PROJECT_SOURCE_DIR}/src
			${PROJECT_SOURCE_DIR}/include
			${PROJECT_BINARY_DIR}/include
			${PROJECT_SOURCE_DIR}/testsuite
			${COIN_TARGET_INCLUDE_DIRECTORIES}
		)
		add_test(NAME VulkanBackendLifecycleTest COMMAND VulkanBackendLifecycleTest)
		set_tests_properties(VulkanBackendLifecycleTest PROPERTIES SKIP_RETURN_CODE 77)

		add_executable(VulkanBackendRenderExternalTest vulkan/vulkan-backend-render-external-test.cpp)
		target_link_libraries(VulkanBackendRenderExternalTest Coin ${COIN_TARGET_LINK_LIBRARIES})
		target_compile_definitions(VulkanBackendRenderExternalTest PRIVATE COIN_INTERNAL)
		target_include_directories(VulkanBackendRenderExternalTest PRIVATE
			${PROJECT_SOURCE_DIR}/src
			${PROJECT_SOURCE_DIR}/include
			${PROJECT_BINARY_DIR}/include
			${PROJECT_SOURCE_DIR}/testsuite
			${COIN_TARGET_INCLUDE_DIRECTORIES}
		)
		add_test(NAME VulkanBackendRenderExternalTest COMMAND VulkanBackendRenderExternalTest)
		set_tests_properties(VulkanBackendRenderExternalTest PROPERTIES SKIP_RETURN_CODE 77)

		add_executable(VulkanBackendTextOverlayTest vulkan/vulkan-backend-text-overlay-test.cpp)
		target_link_libraries(VulkanBackendTextOverlayTest Coin ${COIN_TARGET_LINK_LIBRARIES})
		target_compile_definitions(VulkanBackendTextOverlayTest PRIVATE COIN_INTERNAL)
		target_include_directories(VulkanBackendTextOverlayTest PRIVATE
			${PROJECT_SOURCE_DIR}/src
			${PROJECT_SOURCE_DIR}/include
			${PROJECT_BINARY_DIR}/include
			${PROJECT_SOURCE_DIR}/testsuite
			${COIN_TARGET_INCLUDE_DIRECTORIES}
		)
		add_test(NAME VulkanBackendTextOverlayTest COMMAND VulkanBackendTextOverlayTest)
		set_tests_properties(VulkanBackendTextOverlayTest PROPERTIES SKIP_RETURN_CODE 77)

		add_executable(VulkanBackendMixedTopologyTest vulkan/vulkan-backend-mixed-topology-test.cpp)
		target_link_libraries(VulkanBackendMixedTopologyTest Coin ${COIN_TARGET_LINK_LIBRARIES})
		target_compile_definitions(VulkanBackendMixedTopologyTest PRIVATE COIN_INTERNAL)
		target_include_directories(VulkanBackendMixedTopologyTest PRIVATE
			${PROJECT_SOURCE_DIR}/src
			${PROJECT_SOURCE_DIR}/include
			${PROJECT_BINARY_DIR}/include
			${PROJECT_SOURCE_DIR}/testsuite
			${COIN_TARGET_INCLUDE_DIRECTORIES}
		)
		add_test(NAME VulkanBackendMixedTopologyTest COMMAND VulkanBackendMixedTopologyTest)
		set_tests_properties(VulkanBackendMixedTopologyTest PROPERTIES SKIP_RETURN_CODE 77)

		add_executable(VulkanBackendTransparencyTest vulkan/vulkan-backend-transparency-test.cpp)
		target_link_libraries(VulkanBackendTransparencyTest Coin ${COIN_TARGET_LINK_LIBRARIES})
		target_compile_definitions(VulkanBackendTransparencyTest PRIVATE COIN_INTERNAL)
		target_include_directories(VulkanBackendTransparencyTest PRIVATE
			${PROJECT_SOURCE_DIR}/src
			${PROJECT_SOURCE_DIR}/include
			${PROJECT_BINARY_DIR}/include
			${PROJECT_SOURCE_DIR}/testsuite
			${COIN_TARGET_INCLUDE_DIRECTORIES}
		)
		add_test(NAME VulkanBackendTransparencyTest COMMAND VulkanBackendTransparencyTest)
		set_tests_properties(VulkanBackendTransparencyTest PROPERTIES SKIP_RETURN_CODE 77)

		add_executable(VulkanBackendLightingTest vulkan/vulkan-backend-lighting-test.cpp)
		target_link_libraries(VulkanBackendLightingTest Coin ${COIN_TARGET_LINK_LIBRARIES})
		target_compile_definitions(VulkanBackendLightingTest PRIVATE COIN_INTERNAL)
		target_include_directories(VulkanBackendLightingTest PRIVATE
			${PROJECT_SOURCE_DIR}/src
			${PROJECT_SOURCE_DIR}/include
			${PROJECT_BINARY_DIR}/include
			${PROJECT_SOURCE_DIR}/testsuite
			${COIN_TARGET_INCLUDE_DIRECTORIES}
		)
		add_test(NAME VulkanBackendLightingTest COMMAND VulkanBackendLightingTest)
		set_tests_properties(VulkanBackendLightingTest PROPERTIES SKIP_RETURN_CODE 77)

		add_executable(VulkanBackendDepthTest vulkan/vulkan-backend-depth-test.cpp)
		target_link_libraries(VulkanBackendDepthTest Coin ${COIN_TARGET_LINK_LIBRARIES})
		target_compile_definitions(VulkanBackendDepthTest PRIVATE COIN_INTERNAL)
		target_include_directories(VulkanBackendDepthTest PRIVATE
			${PROJECT_SOURCE_DIR}/src
			${PROJECT_SOURCE_DIR}/include
			${PROJECT_BINARY_DIR}/include
			${PROJECT_SOURCE_DIR}/testsuite
			${COIN_TARGET_INCLUDE_DIRECTORIES}
		)
		add_test(NAME VulkanBackendDepthTest COMMAND VulkanBackendDepthTest)
		set_tests_properties(VulkanBackendDepthTest PROPERTIES SKIP_RETURN_CODE 77)

		add_executable(VulkanBackendFillModeTest vulkan/vulkan-backend-fill-mode-test.cpp)
		target_link_libraries(VulkanBackendFillModeTest Coin ${COIN_TARGET_LINK_LIBRARIES})
		target_compile_definitions(VulkanBackendFillModeTest PRIVATE COIN_INTERNAL)
		target_include_directories(VulkanBackendFillModeTest PRIVATE
			${PROJECT_SOURCE_DIR}/src
			${PROJECT_SOURCE_DIR}/include
			${PROJECT_BINARY_DIR}/include
			${PROJECT_SOURCE_DIR}/testsuite
			${COIN_TARGET_INCLUDE_DIRECTORIES}
		)
		add_test(NAME VulkanBackendFillModeTest COMMAND VulkanBackendFillModeTest)
		set_tests_properties(VulkanBackendFillModeTest PROPERTIES SKIP_RETURN_CODE 77)

		add_executable(VulkanBackendCullingTest vulkan/vulkan-backend-culling-test.cpp)
		target_link_libraries(VulkanBackendCullingTest Coin ${COIN_TARGET_LINK_LIBRARIES})
		target_compile_definitions(VulkanBackendCullingTest PRIVATE COIN_INTERNAL)
		target_include_directories(VulkanBackendCullingTest PRIVATE
			${PROJECT_SOURCE_DIR}/src
			${PROJECT_SOURCE_DIR}/include
			${PROJECT_BINARY_DIR}/include
			${PROJECT_SOURCE_DIR}/testsuite
			${COIN_TARGET_INCLUDE_DIRECTORIES}
		)
		add_test(NAME VulkanBackendCullingTest COMMAND VulkanBackendCullingTest)
		set_tests_properties(VulkanBackendCullingTest PROPERTIES SKIP_RETURN_CODE 77)

		add_executable(VulkanBackendLightVariantsTest vulkan/vulkan-backend-light-variants-test.cpp)
		target_link_libraries(VulkanBackendLightVariantsTest Coin ${COIN_TARGET_LINK_LIBRARIES})
		target_compile_definitions(VulkanBackendLightVariantsTest PRIVATE COIN_INTERNAL)
		target_include_directories(VulkanBackendLightVariantsTest PRIVATE
			${PROJECT_SOURCE_DIR}/src
			${PROJECT_SOURCE_DIR}/include
			${PROJECT_BINARY_DIR}/include
			${PROJECT_SOURCE_DIR}/testsuite
			${COIN_TARGET_INCLUDE_DIRECTORIES}
		)
		add_test(NAME VulkanBackendLightVariantsTest COMMAND VulkanBackendLightVariantsTest)
		set_tests_properties(VulkanBackendLightVariantsTest PROPERTIES SKIP_RETURN_CODE 77)

		add_executable(VulkanBackendStripTopologyTest vulkan/vulkan-backend-strip-topology-test.cpp)
		target_link_libraries(VulkanBackendStripTopologyTest Coin ${COIN_TARGET_LINK_LIBRARIES})
		target_compile_definitions(VulkanBackendStripTopologyTest PRIVATE COIN_INTERNAL)
		target_include_directories(VulkanBackendStripTopologyTest PRIVATE
			${PROJECT_SOURCE_DIR}/src
			${PROJECT_SOURCE_DIR}/include
			${PROJECT_BINARY_DIR}/include
			${PROJECT_SOURCE_DIR}/testsuite
			${COIN_TARGET_INCLUDE_DIRECTORIES}
		)
		add_test(NAME VulkanBackendStripTopologyTest COMMAND VulkanBackendStripTopologyTest)
		set_tests_properties(VulkanBackendStripTopologyTest PROPERTIES SKIP_RETURN_CODE 77)

		add_executable(VulkanBackendTextureTest vulkan/vulkan-backend-texture-test.cpp)
		target_link_libraries(VulkanBackendTextureTest Coin ${COIN_TARGET_LINK_LIBRARIES})
		target_compile_definitions(VulkanBackendTextureTest PRIVATE COIN_INTERNAL)
		target_include_directories(VulkanBackendTextureTest PRIVATE
			${PROJECT_SOURCE_DIR}/src
			${PROJECT_SOURCE_DIR}/include
			${PROJECT_BINARY_DIR}/include
			${PROJECT_SOURCE_DIR}/testsuite
			${COIN_TARGET_INCLUDE_DIRECTORIES}
		)
		add_test(NAME VulkanBackendTextureTest COMMAND VulkanBackendTextureTest)
		set_tests_properties(VulkanBackendTextureTest PROPERTIES SKIP_RETURN_CODE 77)

		add_executable(VulkanBackendAlphaTestTest vulkan/vulkan-backend-alpha-test-test.cpp)
		target_link_libraries(VulkanBackendAlphaTestTest Coin ${COIN_TARGET_LINK_LIBRARIES})
		target_compile_definitions(VulkanBackendAlphaTestTest PRIVATE COIN_INTERNAL)
		target_include_directories(VulkanBackendAlphaTestTest PRIVATE
			${PROJECT_SOURCE_DIR}/src
			${PROJECT_SOURCE_DIR}/include
			${PROJECT_BINARY_DIR}/include
			${PROJECT_SOURCE_DIR}/testsuite
			${COIN_TARGET_INCLUDE_DIRECTORIES}
		)
		add_test(NAME VulkanBackendAlphaTestTest COMMAND VulkanBackendAlphaTestTest)
		set_tests_properties(VulkanBackendAlphaTestTest PROPERTIES SKIP_RETURN_CODE 77)

		add_executable(VulkanBackendMaterialTest vulkan/vulkan-backend-material-test.cpp)
		target_link_libraries(VulkanBackendMaterialTest Coin ${COIN_TARGET_LINK_LIBRARIES})
		target_compile_definitions(VulkanBackendMaterialTest PRIVATE COIN_INTERNAL)
		target_include_directories(VulkanBackendMaterialTest PRIVATE
			${PROJECT_SOURCE_DIR}/src
			${PROJECT_SOURCE_DIR}/include
			${PROJECT_BINARY_DIR}/include
			${PROJECT_SOURCE_DIR}/testsuite
			${COIN_TARGET_INCLUDE_DIRECTORIES}
		)
		add_test(NAME VulkanBackendMaterialTest COMMAND VulkanBackendMaterialTest)
		set_tests_properties(VulkanBackendMaterialTest PROPERTIES SKIP_RETURN_CODE 77)

		add_executable(VulkanBackendDepthFunctionTest vulkan/vulkan-backend-depth-function-test.cpp)
		target_link_libraries(VulkanBackendDepthFunctionTest Coin ${COIN_TARGET_LINK_LIBRARIES})
		target_compile_definitions(VulkanBackendDepthFunctionTest PRIVATE COIN_INTERNAL)
		target_include_directories(VulkanBackendDepthFunctionTest PRIVATE
			${PROJECT_SOURCE_DIR}/src
			${PROJECT_SOURCE_DIR}/include
			${PROJECT_BINARY_DIR}/include
			${PROJECT_SOURCE_DIR}/testsuite
			${COIN_TARGET_INCLUDE_DIRECTORIES}
		)
		add_test(NAME VulkanBackendDepthFunctionTest COMMAND VulkanBackendDepthFunctionTest)
		set_tests_properties(VulkanBackendDepthFunctionTest PROPERTIES SKIP_RETURN_CODE 77)

		add_executable(VulkanBackendSortedOrderTest vulkan/vulkan-backend-sorted-order-test.cpp)
		target_link_libraries(VulkanBackendSortedOrderTest Coin ${COIN_TARGET_LINK_LIBRARIES})
		target_compile_definitions(VulkanBackendSortedOrderTest PRIVATE COIN_INTERNAL)
		target_include_directories(VulkanBackendSortedOrderTest PRIVATE
			${PROJECT_SOURCE_DIR}/src
			${PROJECT_SOURCE_DIR}/include
			${PROJECT_BINARY_DIR}/include
			${PROJECT_SOURCE_DIR}/testsuite
			${COIN_TARGET_INCLUDE_DIRECTORIES}
		)
		add_test(NAME VulkanBackendSortedOrderTest COMMAND VulkanBackendSortedOrderTest)
		set_tests_properties(VulkanBackendSortedOrderTest PROPERTIES SKIP_RETURN_CODE 77)

		add_executable(VulkanBackendTextureModelsTest vulkan/vulkan-backend-texture-models-test.cpp)
		target_link_libraries(VulkanBackendTextureModelsTest Coin ${COIN_TARGET_LINK_LIBRARIES})
		target_compile_definitions(VulkanBackendTextureModelsTest PRIVATE COIN_INTERNAL)
		target_include_directories(VulkanBackendTextureModelsTest PRIVATE
			${PROJECT_SOURCE_DIR}/src
			${PROJECT_SOURCE_DIR}/include
			${PROJECT_BINARY_DIR}/include
			${PROJECT_SOURCE_DIR}/testsuite
			${COIN_TARGET_INCLUDE_DIRECTORIES}
		)
	add_test(NAME VulkanBackendTextureModelsTest COMMAND VulkanBackendTextureModelsTest)
		set_tests_properties(VulkanBackendTextureModelsTest PROPERTIES SKIP_RETURN_CODE 77)

		add_executable(VulkanBackendScissorTest vulkan/vulkan-backend-scissor-test.cpp)
		target_link_libraries(VulkanBackendScissorTest Coin ${COIN_TARGET_LINK_LIBRARIES})
		target_compile_definitions(VulkanBackendScissorTest PRIVATE COIN_INTERNAL)
		target_include_directories(VulkanBackendScissorTest PRIVATE
			${PROJECT_SOURCE_DIR}/src
			${PROJECT_SOURCE_DIR}/include
			${PROJECT_BINARY_DIR}/include
			${PROJECT_SOURCE_DIR}/testsuite
			${COIN_TARGET_INCLUDE_DIRECTORIES}
		)
		add_test(NAME VulkanBackendScissorTest COMMAND VulkanBackendScissorTest)
		set_tests_properties(VulkanBackendScissorTest PROPERTIES SKIP_RETURN_CODE 77)

		add_executable(VulkanBackendBlendingTest vulkan/vulkan-backend-blending-test.cpp)
		target_link_libraries(VulkanBackendBlendingTest Coin ${COIN_TARGET_LINK_LIBRARIES})
		target_compile_definitions(VulkanBackendBlendingTest PRIVATE COIN_INTERNAL)
		target_include_directories(VulkanBackendBlendingTest PRIVATE
			${PROJECT_SOURCE_DIR}/src
			${PROJECT_SOURCE_DIR}/include
			${PROJECT_BINARY_DIR}/include
			${PROJECT_SOURCE_DIR}/testsuite
			${COIN_TARGET_INCLUDE_DIRECTORIES}
		)
		add_test(NAME VulkanBackendBlendingTest COMMAND VulkanBackendBlendingTest)
		set_tests_properties(VulkanBackendBlendingTest PROPERTIES SKIP_RETURN_CODE 77)

		add_executable(VulkanBackendClearTest vulkan/vulkan-backend-clear-test.cpp)
		target_link_libraries(VulkanBackendClearTest Coin ${COIN_TARGET_LINK_LIBRARIES})
		target_compile_definitions(VulkanBackendClearTest PRIVATE COIN_INTERNAL)
		target_include_directories(VulkanBackendClearTest PRIVATE
			${PROJECT_SOURCE_DIR}/src
			${PROJECT_SOURCE_DIR}/include
			${PROJECT_BINARY_DIR}/include
			${PROJECT_SOURCE_DIR}/testsuite
			${COIN_TARGET_INCLUDE_DIRECTORIES}
		)
		add_test(NAME VulkanBackendClearTest COMMAND VulkanBackendClearTest)
		set_tests_properties(VulkanBackendClearTest PROPERTIES SKIP_RETURN_CODE 77)

		add_executable(VulkanBackendYFlipTest vulkan/vulkan-backend-yflip-test.cpp)
		target_link_libraries(VulkanBackendYFlipTest Coin ${COIN_TARGET_LINK_LIBRARIES})
		target_compile_definitions(VulkanBackendYFlipTest PRIVATE COIN_INTERNAL)
		target_include_directories(VulkanBackendYFlipTest PRIVATE
			${PROJECT_SOURCE_DIR}/src
			${PROJECT_SOURCE_DIR}/include
			${PROJECT_BINARY_DIR}/include
			${PROJECT_SOURCE_DIR}/testsuite
			${COIN_TARGET_INCLUDE_DIRECTORIES}
		)
		add_test(NAME VulkanBackendYFlipTest COMMAND VulkanBackendYFlipTest)
		set_tests_properties(VulkanBackendYFlipTest PROPERTIES SKIP_RETURN_CODE 77)

		add_executable(VulkanBackendVertexColorTest vulkan/vulkan-backend-vertex-color-test.cpp)
		target_link_libraries(VulkanBackendVertexColorTest Coin ${COIN_TARGET_LINK_LIBRARIES})
		target_compile_definitions(VulkanBackendVertexColorTest PRIVATE COIN_INTERNAL)
		target_include_directories(VulkanBackendVertexColorTest PRIVATE
			${PROJECT_SOURCE_DIR}/src
			${PROJECT_SOURCE_DIR}/include
			${PROJECT_BINARY_DIR}/include
			${PROJECT_SOURCE_DIR}/testsuite
			${COIN_TARGET_INCLUDE_DIRECTORIES}
		)
		add_test(NAME VulkanBackendVertexColorTest COMMAND VulkanBackendVertexColorTest)
		set_tests_properties(VulkanBackendVertexColorTest PROPERTIES SKIP_RETURN_CODE 77)

		add_executable(VulkanBackendStencilTest vulkan/vulkan-backend-stencil-test.cpp)
		target_link_libraries(VulkanBackendStencilTest Coin ${COIN_TARGET_LINK_LIBRARIES})
		target_compile_definitions(VulkanBackendStencilTest PRIVATE COIN_INTERNAL)
		target_include_directories(VulkanBackendStencilTest PRIVATE
			${PROJECT_SOURCE_DIR}/src
			${PROJECT_SOURCE_DIR}/include
			${PROJECT_BINARY_DIR}/include
			${PROJECT_SOURCE_DIR}/testsuite
			${COIN_TARGET_INCLUDE_DIRECTORIES}
		)
		add_test(NAME VulkanBackendStencilTest COMMAND VulkanBackendStencilTest)
		set_tests_properties(VulkanBackendStencilTest PROPERTIES SKIP_RETURN_CODE 77)

		add_executable(VulkanBackendUniformPackingTest vulkan/vulkan-backend-uniform-packing-test.cpp)
		target_link_libraries(VulkanBackendUniformPackingTest Coin ${COIN_TARGET_LINK_LIBRARIES})
		target_compile_definitions(VulkanBackendUniformPackingTest PRIVATE COIN_INTERNAL)
		target_include_directories(VulkanBackendUniformPackingTest PRIVATE
			${PROJECT_SOURCE_DIR}/src
			${PROJECT_SOURCE_DIR}/include
			${PROJECT_BINARY_DIR}/include
			${PROJECT_SOURCE_DIR}/testsuite
			${COIN_TARGET_INCLUDE_DIRECTORIES}
		)
		add_test(NAME VulkanBackendUniformPackingTest COMMAND VulkanBackendUniformPackingTest)
		set_tests_properties(VulkanBackendUniformPackingTest PROPERTIES SKIP_RETURN_CODE 77)

		add_executable(VulkanRenderManagerTest vulkan/vulkan-rendermanager-test.cpp)
		target_link_libraries(VulkanRenderManagerTest Coin ${COIN_TARGET_LINK_LIBRARIES})
		target_compile_definitions(VulkanRenderManagerTest PRIVATE COIN_INTERNAL)
		target_include_directories(VulkanRenderManagerTest PRIVATE
			${PROJECT_SOURCE_DIR}/src
			${PROJECT_SOURCE_DIR}/include
			${PROJECT_BINARY_DIR}/include
			${PROJECT_SOURCE_DIR}/testsuite
			${COIN_TARGET_INCLUDE_DIRECTORIES}
		)
		add_test(NAME VulkanRenderManagerTest COMMAND VulkanRenderManagerTest)
		set_tests_properties(VulkanRenderManagerTest PROPERTIES SKIP_RETURN_CODE 77)

		add_executable(VulkanLightStabilityTest vulkan/vulkan-light-stability-test.cpp)
		target_link_libraries(VulkanLightStabilityTest Coin ${COIN_TARGET_LINK_LIBRARIES})
		target_compile_definitions(VulkanLightStabilityTest PRIVATE COIN_INTERNAL)
		target_include_directories(VulkanLightStabilityTest PRIVATE
			${PROJECT_SOURCE_DIR}/src
			${PROJECT_SOURCE_DIR}/include
			${PROJECT_BINARY_DIR}/include
			${PROJECT_SOURCE_DIR}/testsuite
			${COIN_TARGET_INCLUDE_DIRECTORIES}
		)
		add_test(NAME VulkanLightStabilityTest COMMAND VulkanLightStabilityTest)
		set_tests_properties(VulkanLightStabilityTest PROPERTIES SKIP_RETURN_CODE 77)
	endif()

	add_executable(EGLBindingTest egl-binding-test.cpp)
	target_link_libraries(EGLBindingTest Coin ${COIN_TARGET_LINK_LIBRARIES})
	target_compile_definitions(EGLBindingTest PRIVATE COIN_INTERNAL)
	target_include_directories(EGLBindingTest PRIVATE
		${PROJECT_SOURCE_DIR}/src
		${PROJECT_SOURCE_DIR}/include
		${PROJECT_BINARY_DIR}/include
		${COIN_TARGET_INCLUDE_DIRECTORIES}
	)
	add_test(NAME EGLBindingTest COMMAND EGLBindingTest)
	set_tests_properties(EGLBindingTest PROPERTIES SKIP_RETURN_CODE 77)

	add_executable(GLSLRuntimeTest glsl-runtime-test.cpp)
	target_link_libraries(GLSLRuntimeTest Coin ${COIN_TARGET_LINK_LIBRARIES})
	target_compile_definitions(GLSLRuntimeTest PRIVATE COIN_INTERNAL)
	target_include_directories(GLSLRuntimeTest PRIVATE
		${PROJECT_SOURCE_DIR}/src
		${PROJECT_SOURCE_DIR}/include
		${PROJECT_BINARY_DIR}/include
		${COIN_TARGET_INCLUDE_DIRECTORIES}
	)
	add_test(NAME GLSLRuntimeTest COMMAND GLSLRuntimeTest)
	set_tests_properties(GLSLRuntimeTest PROPERTIES SKIP_RETURN_CODE 77)

	add_executable(GLSLDiagnosticsTest glsl-diagnostics-test.cpp)
	target_link_libraries(GLSLDiagnosticsTest Coin ${COIN_TARGET_LINK_LIBRARIES})
	target_compile_definitions(GLSLDiagnosticsTest PRIVATE COIN_INTERNAL)
	target_include_directories(GLSLDiagnosticsTest PRIVATE
		${PROJECT_SOURCE_DIR}/src
		${PROJECT_SOURCE_DIR}/include
		${PROJECT_BINARY_DIR}/include
		${COIN_TARGET_INCLUDE_DIRECTORIES}
	)
	add_test(NAME GLSLDiagnosticsTest COMMAND GLSLDiagnosticsTest)
	set_tests_properties(GLSLDiagnosticsTest PROPERTIES SKIP_RETURN_CODE 77)

	add_executable(OffscreenReadbackTest offscreen-readback-test.cpp)
	target_link_libraries(OffscreenReadbackTest Coin ${COIN_TARGET_LINK_LIBRARIES})
	target_compile_definitions(OffscreenReadbackTest PRIVATE COIN_INTERNAL)
	target_include_directories(OffscreenReadbackTest PRIVATE
		${PROJECT_SOURCE_DIR}/src
		${PROJECT_SOURCE_DIR}/include
		${PROJECT_BINARY_DIR}/include
		${COIN_TARGET_INCLUDE_DIRECTORIES}
	)
	add_test(NAME OffscreenReadbackTest COMMAND OffscreenReadbackTest)
	set_tests_properties(OffscreenReadbackTest PROPERTIES SKIP_RETURN_CODE 77)
endif()

# Many warnings are generated from test macros on macOS with Xcode.
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG(-Wtautological-compare DISABLE_WARNING_TAUTOLOGCOMPARE)
if(DISABLE_WARNING_TAUTOLOGCOMPARE)
	target_compile_options(CoinTests PUBLIC -Wno-tautological-compare)
endif()
