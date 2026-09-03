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

#include "shaders/SoGLSLShaderObject.h"
#include <Inventor/elements/SoMultiTextureImageElement.h>
#include "coindefs.h"

#include <Inventor/system/gl.h>
#include <cassert>
#include <cstdio>
#include <Inventor/errors/SoDebugError.h>

// GL 3.2+ shader types not in macOS legacy GL headers
#ifdef __APPLE__
#ifndef GL_GEOMETRY_SHADER
#define GL_GEOMETRY_SHADER 0x8DD9
#endif
#endif

#include "glue/glp.h"
#include "glue/glslp.h"
#include "rendering/SoGL.h"
#include "shaders/SoGLSLShaderDiagnostics.h"
#include "shaders/SoGLSLShaderParameter.h"

static int32_t soglshaderobject_idcounter = 1;

SoGLSLShaderObject::SoGLSLShaderObject(const uint32_t cachecontext)
  : SoGLShaderObject(cachecontext)
{
  this->programHandle = 0;
  this->shaderHandle = 0;
  this->isattached = FALSE;
  this->programid = 0;
}

SoGLSLShaderObject::~SoGLSLShaderObject()
{
  // make sure we don't detach, since the program might have been
  // destructed already. FIXME: investigate if not calling detach will
  // lead to memory leaks. pederb, 2006-10-17
  this->isattached = FALSE;
  this->unload();
}

// *************************************************************************

SoShader::Type
SoGLSLShaderObject::shaderType(void) const
{
  return SoShader::GLSL_SHADER;
}

SbBool
SoGLSLShaderObject::isLoaded(void) const
{
  return (this->shaderHandle != 0);
}

void
SoGLSLShaderObject::load(const char* srcStr)
{
  this->unload();
  this->setParametersDirty(TRUE);

  GLint flag;
  GLenum sType;

  switch (this->getShaderType()) {
  default:
    assert(0 &&" unknown shader type");
  case VERTEX:
    sType = GL_VERTEX_SHADER;
    break;
  case FRAGMENT:
    sType = GL_FRAGMENT_SHADER;
    break;
  case GEOMETRY:
    sType = GL_GEOMETRY_SHADER;
    break;
  }

  SoGLSLShaderObject::didOpenGLErrorOccur("SoGLSLShaderObject::load() : previous errors");

  this->shaderHandle = (COIN_GLhandle) cc_glglue_glCreateShader(this->glctx, sType);
  this->programid = 0;

  if (this->shaderHandle == 0) return;
  this->programid = soglshaderobject_idcounter++;

  const char * preparedSourcePtr = srcStr;
  cc_glglue_glShaderSource(this->glctx, (GLuint) this->shaderHandle, 1,
                           &preparedSourcePtr, NULL);
  cc_glglue_glCompileShader(this->glctx, (GLuint) this->shaderHandle);

  if (SoGLSLShaderObject::didOpenGLErrorOccur("SoGLSLShaderObject::load()")) {
    this->shaderHandle = 0;
    return;
  }

  cc_glglue_glGetShaderiv(this->glctx, (GLuint) this->shaderHandle,
                          GL_COMPILE_STATUS, &flag);
  SoGLSLShaderObject::printInfoLog(this->GLContext(),
                                   this->shaderHandle,
                                   this->getShaderType(),
                                   this->sourceHint,
                                   !flag);

  if (!flag) {
    this->shaderHandle = 0;
  }
}

void
SoGLSLShaderObject::unload(void)
{
  this->detach();
  if (this->shaderHandle) {
    cc_glglue_glDeleteShader(this->glctx, (GLuint) this->shaderHandle);
  }
  this->shaderHandle = 0;
  this->programHandle = 0;
  this->programid = 0;
}

SoGLShaderParameter *
SoGLSLShaderObject::getNewParameter(void) const
{
  return new SoGLSLShaderParameter();
}

// *************************************************************************

void
SoGLSLShaderObject::attach(COIN_GLhandle programHandle)
{
  if (programHandle <= 0 || this->programHandle == programHandle) return;

  detach();

  if (this->shaderHandle) {
    this->programHandle = programHandle;
    cc_glglue_glAttachShader(this->glctx,
                             (GLuint) this->programHandle,
                             (GLuint) this->shaderHandle);
    this->isattached = TRUE;
  }
}

void
SoGLSLShaderObject::detach(void)
{
  if (this->isattached && this->programHandle && this->shaderHandle) {
    cc_glglue_glDetachShader(this->glctx,
                             (GLuint) this->programHandle,
                             (GLuint) this->shaderHandle);
    this->isattached = FALSE;
    this->programHandle = 0;
  }
}

SbBool
SoGLSLShaderObject::isAttached(void) const
{
  return this->isattached;
}

void
SoGLSLShaderObject::printInfoLog(const cc_glglue * g,
                                 COIN_GLhandle handle,
                                 const ShaderType shaderType,
                                 const SbString & sourceHint,
                                 const SbBool failed)
{
  const char * sourceName = sourceHint.getLength() > 0 ?
    sourceHint.getString() : "<unnamed>";
  const SbString infoLog = soglsl_get_info_log(g, (GLuint) handle, FALSE);

  if (infoLog.getLength() > 0) {
    if (failed) {
      SoDebugError::postWarning("SoGLSLShaderObject::printInfoLog",
                                "%s '%s' failed to compile: %s",
                                soglsl_stage_name(shaderType),
                                sourceName,
                                infoLog.getString());
    }
    else {
      SoDebugError::postInfo("SoGLSLShaderObject::printInfoLog",
                             "%s '%s' log: %s",
                             soglsl_stage_name(shaderType),
                             sourceName,
                             infoLog.getString());
    }
  }
  else if (failed) {
    SoDebugError::postWarning("SoGLSLShaderObject::printInfoLog",
                              "%s '%s' failed to compile with no compiler log",
                              soglsl_stage_name(shaderType),
                              sourceName);
  }
}

SbBool
SoGLSLShaderObject::didOpenGLErrorOccur(const SbString & source)
{
  SbBool retCode = FALSE;
  SbBool glerror_debug = sogl_glerror_debugging();

  // only do a glFlush if COIN_GLERROR_DEBUGGING is set since it can
  // degrade performance a lot. If glFlush is not executed here, gl
  // errors from the shaders might not get caught until after the
  // geometry is rendered, which makes debugging really confusing.
  if (glerror_debug) {
    glFlush();
  }

  GLenum glErr = glGetError();
  while (glErr != GL_NO_ERROR) {
    SoDebugError::post(source.getString(), "error: '%s' %s",
                       coin_glerror_string(glErr),
                       glerror_debug ? "":
                       "(set envvar COIN_GLERROR_DEBUGGING=1 "
                       "and re-run to get more information)");

    retCode = TRUE;
    glErr = glGetError();
  }
  return retCode;
}

#include <cstdio>
#include <Inventor/SbName.h>
#include <Inventor/nodes/SoShaderParameter.h>
#if COIN_BUILD_LEGACY_GL_RENDERER
#include <Inventor/elements/SoGLMultiTextureImageElement.h>
#endif
#include <Inventor/elements/SoLightModelElement.h>
#include <Inventor/actions/SoAction.h>
#include <cstdio>

void
SoGLSLShaderObject::updateCoinParameter(SoState * COIN_UNUSED_ARG(state), const SbName & name, SoShaderParameter * param, const int value)
{
  COIN_GLhandle pHandle = this->programHandle;
  if (pHandle) {
    const cc_glglue * glue = this->GLContext();

    // FIXME: set up a dict for the supported Coin variables
    SoShaderParameter1i * p = (SoShaderParameter1i*) param;

    if (p) {
      if (p->value.getValue() != value) p->value = value;
    }
    else {
      GLint location = cc_glglue_glGetUniformLocation(
        glue, (GLuint) pHandle, name.getString());

#if 0
      fprintf(stderr,"action: %s, name: %s, loc: %d, handle: %p\n",
              state->getAction()->getTypeId().getName().getString(),
              name.getString(), location, pHandle);
#endif
      if (location >= 0) {
        glue->glUniform1i(location, value);
      }
    }
  }
}
