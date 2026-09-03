configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Inventor/C/basic.h.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/Inventor/C/basic.h)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Inventor/system/gl-headers.h.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/Inventor/system/gl-headers.h)

# Make the namespaced Khronos snapshot available to build-tree consumers as
# well as installed consumers. This is needed by tests and examples that
# include Coin's generated public headers before the install step.
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/Inventor/system/khronos)
configure_file(${PROJECT_SOURCE_DIR}/src/glue/khronos/GL/glcorearb.h
  ${CMAKE_CURRENT_BINARY_DIR}/Inventor/system/khronos/glcorearb.h COPYONLY)
configure_file(${PROJECT_SOURCE_DIR}/src/glue/khronos/KHR/khrplatform.h
  ${CMAKE_CURRENT_BINARY_DIR}/Inventor/system/khronos/khrplatform.h COPYONLY)

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
