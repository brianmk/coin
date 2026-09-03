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

// OpenGL context and rendering-state helpers shared by both renderer
// configurations.  The fixed-function drawing helpers live in SoGL.cpp,
// which is only built when the legacy renderer is enabled.

#include "rendering/SoGL.h"
#include "coindefs.h"

#include <cstdlib>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif // HAVE_CONFIG_H

#include <Inventor/C/tidbits.h>
#include <Inventor/misc/SoState.h>

#include "glue/glp.h"

#if COIN_BUILD_LEGACY_GL_RENDERER
#include <Inventor/actions/SoGLRenderAction.h>
#include <Inventor/errors/SoDebugError.h>
#endif

// Convenience function for access to OpenGL wrapper from an SoState
// pointer.
const cc_glglue *
sogl_glue_instance(const SoState * state)
{
#if COIN_BUILD_LEGACY_GL_RENDERER
  SoGLRenderAction * action = (SoGLRenderAction *)state->getAction();
  // FIXME: disabled until we figure out why this doesn't work on some
  // Linux systems (gcc 3.2 systems, it seems). pederb, 2003-11-24
#if 0
  assert(action->isOfType(SoGLRenderAction::getClassTypeId()) &&
         "must have state from SoGLRenderAction to get hold of GL wrapper");
  return cc_glglue_instance(action->getCacheContext());
#else // disabled
  if (action->isOfType(SoGLRenderAction::getClassTypeId())) {
    return cc_glglue_instance(action->getCacheContext());
  }
  static int didwarn = 0;
  if (!didwarn) {
    didwarn = 1;
    SoDebugError::postWarning("sogl_glue_instance",
                              "Wrong action type detected. Please report this to <coin-support@coin3d.org>, "
                              "and include information about your system (compiler, Linux version, etc.");
  }
  // just return some cc_glglue instance. It usually doesn't matter
  // that much unless multiple contexts on multiple displays are used.
  return cc_glglue_instance(1);
#endif // workaround version
#else
  // Core-only code can still use the glue for context-independent queries,
  // but there is no SoGLRenderAction from which to obtain a context id.
  (void)state;
  return cc_glglue_instance(1);
#endif
}

SbBool
sogl_context_supports_legacy_rendering(const SoState * state)
{
  const cc_glglue * glue = sogl_glue_instance(state);
  return cc_glglue_context_supports_legacy_rendering(glue);
}

// Used by library code to decide whether or not to add extra
// debugging checks for glGetError().
SbBool
sogl_glerror_debugging(void)
{
  static int COIN_GLERROR_DEBUGGING = -1;
  if (COIN_GLERROR_DEBUGGING == -1) {
    const char * str = coin_getenv("COIN_GLERROR_DEBUGGING");
    COIN_GLERROR_DEBUGGING = str ? atoi(str) : 0;
  }
  return (COIN_GLERROR_DEBUGGING == 0) ? FALSE : TRUE;
}
