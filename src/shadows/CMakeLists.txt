# source files
set(COIN_SHADOWS_FILES)

if(COIN_BUILD_LEGACY_GL_RENDERER)
	list(APPEND COIN_SHADOWS_FILES
		SoShadowGroup.cpp
		SoShadowStyle.cpp
		SoShadowSpotLight.cpp
		SoShadowDirectionalLight.cpp
		SoShadowStyleElement.cpp
		SoShadowCulling.cpp
		SoGLShadowCullingElement.cpp
	)
endif()

# build library
if(COIN_BUILD_SINGLE_LIB)
	foreach(SRCFILE ${COIN_SHADOWS_FILES})
		set(COIN_SOURCE ${COIN_SOURCE} shadows/${SRCFILE} CACHE INTERNAL "COIN_SOURCE")
	endforeach(SRCFILE)
elseif(COIN_SHADOWS_FILES)
	add_library(shadows OBJECT ${COIN_SHADOWS_FILES})
endif()

# define documentation
foreach(SRCFILE ${COIN_SHADOWS_FILES})
	set(COIN_DOCUMENTATION_FILES ${COIN_DOCUMENTATION_FILES} ${PROJECT_SOURCE_DIR}/src/shadows/${SRCFILE} CACHE INTERNAL "COIN_DOCUMENTATION_FILES")
endforeach(SRCFILE)
