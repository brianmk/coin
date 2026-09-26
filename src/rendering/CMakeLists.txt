# source files
set(COIN_RENDERING_FILES
	gl/SoGLContext.cpp
	gl/SoGLDriverDatabase.cpp
	gl/SoGLNurbs.cpp
	gl/CoinOffscreenGLCanvas.cpp
	backend/SoRenderManager.cpp
	backend/SoRenderManagerP.cpp
	backend/SoRenderIR.cpp
	backend/SoRenderBackend.cpp
	# SoGLRenderBackend.cpp is intentionally NOT part of libCoin: it is a
	# reference DrawList consumer used only by the testsuite. The production
	# OpenGL viewport renders through SoRenderManager/SoGLRenderAction, not the
	# IR, so the class is compiled into the GL testsuite targets instead (see
	# testsuite/CMakeLists.txt). This keeps the installed renderer surface the
	# Vulkan path only.
)

if(COIN_BUILD_VULKAN_RENDERER)
	list(APPEND COIN_RENDERING_FILES
		vulkan/common/core/SoVulkanConfig.cpp
		vulkan/common/timing/SoVulkanGpuTimers.cpp
		vulkan/common/memory/SoVulkanBufferFactory.cpp
		vulkan/common/memory/SoVulkanFrameRing.cpp
		vulkan/common/caches/SoVulkanTextureCache.cpp
		vulkan/common/caches/SoVulkanSamplerCache.cpp
		vulkan/common/memory/SoVulkanStagingPool.cpp
		vulkan/common/memory/SoVulkanGeometryArena.cpp
		vulkan/common/caches/SoVulkanGeometryCache.cpp
		vulkan/common/caches/SoVulkanPipelineCache.cpp
		vulkan/raster/SoVulkanRenderBackendCore.cpp
		vulkan/raster/SoVulkanRenderPassCache.cpp
		vulkan/raster/SoVulkanRenderBackendPipeline.cpp
		vulkan/raster/SoVulkanRenderBackendGeometry.cpp
		vulkan/raster/SoVulkanRenderBackendTexture.cpp
		vulkan/raster/SoVulkanRenderBackendCommand.cpp
		vulkan/raster/SoVulkanRenderBackendWideLine.cpp
		vulkan/raster/SoVulkanRenderBackendGeometryLod.cpp
		vulkan/raster/SoVulkanRenderBackendFrame.cpp
		vulkan/raster/SoVulkanRenderBackendOutput.cpp
		vulkan/manager/SoVulkanRenderManager.cpp
		vulkan/manager/SoVulkanRenderManagerRender.cpp
		vulkan/raytracing/rtx/SoRTXRenderBackendCore.cpp
		vulkan/raytracing/rtx/SoRTXRenderBackendPipeline.cpp
		vulkan/raytracing/rtx/SoRTXRenderBackendGeometry.cpp
		vulkan/raytracing/rtx/SoRTXRenderBackendBuffers.cpp
		vulkan/raytracing/rtx/SoRTXRenderBackendTextures.cpp
		vulkan/raytracing/rtx/SoRTXRenderBackendPathTracing.cpp
		vulkan/raytracing/denoise/SoRTXRenderBackendDenoise.cpp
		vulkan/raytracing/denoise/SoRTXRenderBackendRtx.cpp
		vulkan/raytracing/denoise/SoRTXRenderBackendDnsr.cpp
		vulkan/raytracing/rtx/SoRTXRenderBackendPick.cpp
		third_party/vma/vk_mem_alloc_impl.cpp
	)
endif()

if(COIN_BUILD_LEGACY_GL_RENDERER)
	list(APPEND COIN_RENDERING_FILES
		gl/SoGL.cpp
		gl/SoGLBigImage.cpp
		gl/SoGLImage.cpp
		gl/SoGLCubeMapImage.cpp
		gl/SoOffscreenRenderer.cpp
		gl/SoOffscreenCGData.cpp
		gl/SoOffscreenGLXData.cpp
		gl/SoOffscreenWGLData.cpp
		# These belong to the legacy renderer's resource/draw architecture.
		# VBOs themselves are not unsupported by core OpenGL.
		gl/SoVBO.cpp
		gl/SoVertexArrayIndexer.cpp
	)
endif()

# Files excluded from public API documentation, included in complete documentation.
set(COIN_RENDERING_INTERNAL_FILES
	gl/SoGL.h
	gl/SoGL.cpp
	gl/SoGLContext.cpp
	gl/SoGLNurbs.h
	gl/SoGLNurbs.cpp
	backend/SoRenderManagerP.h
	backend/SoRenderManagerP.cpp
	gl/SoOffscreenCGData.h
	gl/SoOffscreenCGData.cpp
	gl/SoOffscreenGLXData.h
	gl/SoOffscreenGLXData.cpp
	gl/SoOffscreenWGLData.h
	gl/SoOffscreenWGLData.cpp
	gl/SoVBO.h
	gl/SoVBO.cpp
	gl/SoVertexArrayIndexer.h
	gl/SoVertexArrayIndexer.cpp
	gl/CoinOffscreenGLCanvas.h
	gl/CoinOffscreenGLCanvas.cpp
	backend/SoRenderIRP.h
	backend/SoRenderBackend.h
	gl/SoGLRenderBackend.h
	vulkan/common/core/SoVulkanConfig.h
	vulkan/common/core/SoVulkanResult.h
	vulkan/common/memory/SoVulkanBufferFactory.h
	vulkan/common/memory/SoVulkanFrameRing.h
	vulkan/common/caches/SoVulkanTextureCache.h
	vulkan/common/caches/SoVulkanSamplerCache.h
	vulkan/common/memory/SoVulkanStagingPool.h
	vulkan/common/memory/SoVulkanGeometryArena.h
	vulkan/common/caches/SoVulkanGeometryCache.h
	vulkan/common/caches/SoVulkanPipelineCache.h
	vulkan/raster/SoVulkanRenderBackend.h
	vulkan/raster/SoVulkanRenderBackendP.h
	vulkan/raster/SoVulkanRenderPassCache.h
	vulkan/common/core/SoVulkanShared.h
	vulkan/common/core/SoVulkanVma.h
	vulkan/raytracing/rtx/SoRTXRenderBackend.h
	vulkan/raytracing/rtx/SoRTXRenderBackendP.h
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
