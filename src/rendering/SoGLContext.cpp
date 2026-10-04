// OpenGL context/state helpers shared by both renderer configurations.
// The fixed-function drawing helpers live in SoGL.cpp (legacy renderer only).

#include "rendering/SoGL.h"
#include "coindefs.h"

#include <cstdlib>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif // HAVE_CONFIG_H

#include <Inventor/C/tidbits.h>
#include <Inventor/misc/SoState.h>

#include "glue/glp.h"

#include <Inventor/actions/SoGLRenderAction.h>
#include <Inventor/errors/SoDebugError.h>

// Convenience function for access to OpenGL wrapper from an SoState
// pointer.
const cc_glglue *
sogl_glue_instance(const SoState * state)
{
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
