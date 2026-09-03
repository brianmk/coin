# source files
set(COIN_GLELEMENTS_FILES
	SoGLCacheContextElement.cpp
	SoGLClipPlaneElement.cpp
	SoGLColorIndexElement.cpp
	SoGLCoordinateElement.cpp
	SoGLDepthBufferElement.cpp
	SoGLDrawStyleElement.cpp
	SoGLEnvironmentElement.cpp
	SoGLLazyElement.cpp
	SoGLLightIdElement.cpp
	SoGLLinePatternElement.cpp
	SoGLLineWidthElement.cpp
	SoGLModelMatrixElement.cpp
	SoGLMultiTextureCoordinateElement.cpp
	SoGLMultiTextureEnabledElement.cpp
	SoGLMultiTextureMatrixElement.cpp
	SoGLMultiTextureImageElement.cpp
	SoGLNormalElement.cpp
	SoGLPointSizeElement.cpp
	SoGLPolygonOffsetElement.cpp
	SoGLProjectionMatrixElement.cpp
	SoGLRenderPassElement.cpp
	SoGLShapeHintsElement.cpp
	SoGLUpdateAreaElement.cpp
	SoGLVBOElement.cpp
	SoGLViewingMatrixElement.cpp
	SoGLViewportRegionElement.cpp
	SoGLDisplayList.cpp
	SoGLVertexAttributeElement.cpp
	SoResetMatrixElement.cpp
)

# SoGLCacheContextElement is shared by the EGL/offscreen and shader paths.
# The remaining GL elements implement the legacy renderer and are omitted from
# core-only builds.
set(COIN_CORE_GLELEMENTS_FILES
	SoGLCacheContextElement.cpp
)

if(COIN_BUILD_SINGLE_LIB)
	foreach(SRCFILE ${COIN_CORE_GLELEMENTS_FILES})
		set(COIN_SOURCE
			${COIN_SOURCE}
			elements/GL/${SRCFILE}
			CACHE INTERNAL "COIN_SOURCE"
		)
	endforeach(SRCFILE)
	if(COIN_BUILD_LEGACY_GL_RENDERER)
		set(COIN_LEGACY_GLELEMENTS_FILES ${COIN_GLELEMENTS_FILES})
		list(REMOVE_ITEM COIN_LEGACY_GLELEMENTS_FILES ${COIN_CORE_GLELEMENTS_FILES})
		foreach(SRCFILE ${COIN_LEGACY_GLELEMENTS_FILES})
			set(COIN_SOURCE
				${COIN_SOURCE}
				elements/GL/${SRCFILE}
				CACHE INTERNAL "COIN_SOURCE"
			)
		endforeach()
	endif()
else()
	add_library(glelements OBJECT ${COIN_CORE_GLELEMENTS_FILES})
	if(COIN_BUILD_LEGACY_GL_RENDERER)
		set(COIN_LEGACY_GLELEMENTS_FILES ${COIN_GLELEMENTS_FILES})
		list(REMOVE_ITEM COIN_LEGACY_GLELEMENTS_FILES ${COIN_CORE_GLELEMENTS_FILES})
		target_sources(glelements PRIVATE ${COIN_LEGACY_GLELEMENTS_FILES})
	endif()
endif()

# define documentation
foreach(SRCFILE ${COIN_GLELEMENTS_FILES})
	set(COIN_DOCUMENTATION_FILES ${COIN_DOCUMENTATION_FILES} ${PROJECT_SOURCE_DIR}/src/elements/GL/${SRCFILE} CACHE INTERNAL "COIN_DOCUMENTATION_FILES")
endforeach(SRCFILE)
set(COIN_DOCUMENTATION_FILES
	${COIN_DOCUMENTATION_FILES}
	${PROJECT_SOURCE_DIR}/src/elements/GL/SoResetMatrixElement.h
	CACHE INTERNAL "COIN_DOCUMENTATION_FILES"
)
