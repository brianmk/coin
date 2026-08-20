# source files
set(COIN_RENDERING_FILES
	SoGLContext.cpp
	SoGLDriverDatabase.cpp
	SoGLNurbs.cpp
	CoinOffscreenGLCanvas.cpp
	SoRenderManager.cpp
	SoRenderManagerP.cpp
	SoRenderIR.cpp
	SoRenderBackend.cpp
	SoGLRenderBackend.cpp
)

if(COIN_BUILD_VULKAN_RENDERER)
	list(APPEND COIN_RENDERING_FILES
		SoVulkanRenderBackend.cpp
		SoVulkanRenderManager.cpp
		SoRTXRenderBackend.cpp
	)
endif()

if(COIN_BUILD_LEGACY_GL_RENDERER)
	list(APPEND COIN_RENDERING_FILES
		SoGL.cpp
		SoGLBigImage.cpp
		SoGLImage.cpp
		SoGLCubeMapImage.cpp
		SoOffscreenRenderer.cpp
		SoOffscreenCGData.cpp
		SoOffscreenGLXData.cpp
		SoOffscreenWGLData.cpp
		# These belong to the legacy renderer's resource/draw architecture.
		# VBOs themselves are not unsupported by core OpenGL.
		SoVBO.cpp
		SoVertexArrayIndexer.cpp
	)
endif()

# Files excluded from public API documentation, included in complete documentation.
set(COIN_RENDERING_INTERNAL_FILES
	SoGL.h
	SoGL.cpp
	SoGLContext.cpp
	SoGLNurbs.h
	SoGLNurbs.cpp
	SoRenderManagerP.h
	SoRenderManagerP.cpp
	SoOffscreenCGData.h
	SoOffscreenCGData.cpp
	SoOffscreenGLXData.h
	SoOffscreenGLXData.cpp
	SoOffscreenWGLData.h
	SoOffscreenWGLData.cpp
	SoVBO.h
	SoVBO.cpp
	SoVertexArrayIndexer.h
	SoVertexArrayIndexer.cpp
	CoinOffscreenGLCanvas.h
	CoinOffscreenGLCanvas.cpp
	SoRenderIRP.h
	SoRenderBackend.h
	SoGLRenderBackend.h
	SoVulkanRenderBackend.h
	SoVulkanRenderManager.h
	SoRTXRenderBackend.h
)

# build library
if(COIN_BUILD_SINGLE_LIB)
	foreach(SRCFILE ${COIN_RENDERING_FILES})
		set(COIN_SOURCE ${COIN_SOURCE} rendering/${SRCFILE} CACHE INTERNAL "COIN_SOURCE")
	endforeach(SRCFILE)
else()
	add_library(rendering OBJECT ${COIN_RENDERING_FILES})
endif()

# define documentation
foreach(SRCFILE ${COIN_RENDERING_FILES})
	set(COIN_DOCUMENTATION_FILES ${COIN_DOCUMENTATION_FILES} ${PROJECT_SOURCE_DIR}/src/rendering/${SRCFILE} CACHE INTERNAL "COIN_DOCUMENTATION_FILES")
endforeach(SRCFILE)
foreach(SRCFILE ${COIN_RENDERING_INTERNAL_FILES})
	set(COIN_INTERNAL_DOCUMENTATION_FILES ${COIN_INTERNAL_DOCUMENTATION_FILES} ${PROJECT_SOURCE_DIR}/src/rendering/${SRCFILE} CACHE INTERNAL "COIN_INTERNAL_DOCUMENTATION_FILES")
endforeach(SRCFILE)
