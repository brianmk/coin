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
if(HAVE_EGL)
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
