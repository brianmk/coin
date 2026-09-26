/**************************************************************************\
 * Copyright (c) Kongsberg Oil & Gas Technologies AS
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 * 
 * Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 
 * Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 * 
 * Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
\**************************************************************************/

#include "gl/CoinOffscreenGLCanvas.cpp"
#include "gl/SoGLContext.cpp"
#if COIN_BUILD_LEGACY_GL_RENDERER
#include "gl/SoGL.cpp"
#include "gl/SoGLBigImage.cpp"
#include "gl/SoGLCubeMapImage.cpp"
#include "gl/SoGLImage.cpp"
#include "gl/SoGLNurbs.cpp"
#include "gl/SoOffscreenCGData.cpp"
#include "gl/SoOffscreenGLXData.cpp"
#include "gl/SoOffscreenRenderer.cpp"
#include "gl/SoOffscreenWGLData.cpp"
#include "gl/SoVBO.cpp"
#include "gl/SoVertexArrayIndexer.cpp"
#endif
#include "backend/SoRenderManager.cpp"
#include "backend/SoRenderManagerP.cpp"
#include "gl/SoGLDriverDatabase.cpp"
#include "backend/SoRenderIR.cpp"
#include "backend/SoRenderBackend.cpp"
#include "gl/SoGLRenderBackend.cpp"
