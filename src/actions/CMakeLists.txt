# source files
set(COIN_ACTIONS_FILES
	SoAction.cpp
	SoActionP.cpp
	SoCallbackAction.cpp
	SoGetBoundingBoxAction.cpp
	SoGetMatrixAction.cpp
	SoGetPrimitiveCountAction.cpp
	SoHandleEventAction.cpp
	SoPickAction.cpp
	SoRayPickAction.cpp
	SoSearchAction.cpp
	SoSimplifyAction.cpp
	SoToVRMLAction.cpp
	SoToVRML2Action.cpp
	SoWriteAction.cpp
	SoAudioRenderAction.cpp
	SoIRRenderAction.cpp
)

if(COIN_BUILD_LEGACY_GL_RENDERER)
	list(APPEND COIN_ACTIONS_FILES
		SoBoxHighlightRenderAction.cpp
		SoGLRenderAction.cpp
		SoLineHighlightRenderAction.cpp
		# Reorganization currently depends on the LegacyGL primitive cache.
		SoReorganizeAction.cpp
	)
endif()

# Files excluded from public API documentation, included in complete documentation.
set(COIN_ACTIONS_INTERNAL_FILES
	SoActionP.h
	SoActionP.cpp
	SoSubActionP.h
)

# build library
if(COIN_BUILD_SINGLE_LIB)
	foreach(SRCFILE ${COIN_ACTIONS_FILES})
		set(COIN_SOURCE ${COIN_SOURCE} actions/${SRCFILE} CACHE INTERNAL "COIN_SOURCE")
	endforeach(SRCFILE)
else()
	add_library(actions OBJECT ${COIN_ACTIONS_FILES})
endif()

# define documentation
foreach(SRCFILE ${COIN_ACTIONS_FILES})
	set(COIN_DOCUMENTATION_FILES ${COIN_DOCUMENTATION_FILES} ${PROJECT_SOURCE_DIR}/src/actions/${SRCFILE} CACHE INTERNAL "COIN_DOCUMENTATION_FILES")
endforeach(SRCFILE)
foreach(SRCFILE ${COIN_ACTIONS_INTERNAL_FILES})
	set(COIN_INTERNAL_DOCUMENTATION_FILES ${COIN_INTERNAL_DOCUMENTATION_FILES} ${PROJECT_SOURCE_DIR}/src/actions/${SRCFILE} CACHE INTERNAL "COIN_INTERNAL_DOCUMENTATION_FILES")
endforeach(SRCFILE)
