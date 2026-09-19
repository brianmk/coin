configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Inventor/C/basic.h.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/Inventor/C/basic.h)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Inventor/system/gl-headers.h.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/Inventor/system/gl-headers.h)

# Make the namespaced Khronos snapshot available to build-tree consumers as
# well as installed consumers. This is needed by tests and examples that
# include Coin's generated public headers before the install step.
set(COIN_KHRONOS_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/Inventor/system/khronos)
file(MAKE_DIRECTORY ${COIN_KHRONOS_INCLUDE_DIR})
# The vendored glcorearb.h pulls in its platform types with
# "#include <KHR/khrplatform.h>", which only resolves while Coin's own source
# tree (src/glue/khronos) is on the include path. The namespaced snapshot
# keeps khrplatform.h beside glcorearb.h and is consumed without that path, so
# rewrite the include to a relative one here. The vendored file itself stays
# pristine; see src/glue/khronos/PROVENANCE.txt.
file(READ ${PROJECT_SOURCE_DIR}/src/glue/khronos/GL/glcorearb.h _coin_glcorearb)
string(REPLACE
  "#include <KHR/khrplatform.h>"
  "#include \"khrplatform.h\""
  _coin_glcorearb "${_coin_glcorearb}")
file(WRITE ${COIN_KHRONOS_INCLUDE_DIR}/glcorearb.h "${_coin_glcorearb}")
configure_file(${PROJECT_SOURCE_DIR}/src/glue/khronos/KHR/khrplatform.h
  ${COIN_KHRONOS_INCLUDE_DIR}/khrplatform.h COPYONLY)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  ${PROJECT_SOURCE_DIR}/src/glue/khronos/GL/glcorearb.h)
set(COIN_KHRONOS_INCLUDE_DIR ${COIN_KHRONOS_INCLUDE_DIR} PARENT_SCOPE)

file(GLOB_RECURSE COMMON_HDRS ${CMAKE_CURRENT_SOURCE_DIR}/Inventor/*.h)
file(GLOB_RECURSE COMMON_BIN_HDRS ${CMAKE_CURRENT_BINARY_DIR}/Inventor/*.h)

set(COIN_DOCUMENTATION_FILES
  ${COIN_DOCUMENTATION_FILES}
  ${CMAKE_CURRENT_SOURCE_DIR}/SoDebug.h
  ${CMAKE_CURRENT_BINARY_DIR}/Inventor/C/basic.h
  ${COMMON_HDRS}
  CACHE INTERNAL "COIN_DOCUMENTATION_FILES"
)
set(COIN_INTERNAL_DOCUMENTATION_FILES
  ${COIN_INTERNAL_DOCUMENTATION_FILES}
  ${CMAKE_CURRENT_SOURCE_DIR}/Inventor/C/glue/gl.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Inventor/C/glue/spidermonkey.h
  CACHE INTERNAL "COIN_INTERNAL_DOCUMENTATION_FILES"
)

set(COMMON_HDRS ${COMMON_HDRS} ${COMMON_BIN_HDRS} PARENT_SCOPE)
