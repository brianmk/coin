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

/*
 *  Environment variable controls available:
 * 
 *   - COIN_EGL_CORE_PROFILE: set to 1 to request an OpenGL core-profile
 *     context for offscreen rendering.
 */

#include "glue/gl_egl.h"
#include "coindefs.h"

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif /* HAVE_CONFIG_H */

#include <cstdlib>
#include <cstring>
#include <cassert>
#include <climits>

#include <Inventor/C/tidbits.h>
#include <Inventor/C/glue/gl.h>
#include <Inventor/C/errors/debugerror.h>
#include <Inventor/C/glue/dl.h>

#include "glue/glp.h"
#include "glue/dlp.h"

/* ********************************************************************** */

#ifndef HAVE_EGL

void * eglglue_getprocaddress(const cc_glglue * glue_in, const char * fname)
{
  return NULL;
}

void * eglglue_context_create_offscreen(unsigned int COIN_UNUSED_ARG(width),
                                        unsigned int COIN_UNUSED_ARG(height)) {
  return NULL;
}

SbBool eglglue_context_make_current(void * COIN_UNUSED_ARG(ctx))
{
  return FALSE;
}

void eglglue_context_reinstate_previous(void * COIN_UNUSED_ARG(ctx))
{
  assert(FALSE);
}

void eglglue_context_destruct(void * COIN_UNUSED_ARG(ctx))
{
  assert(FALSE);
}

SbBool eglglue_context_pbuffer_max(void * ctx, unsigned int * lims)
{
  assert(FALSE); return FALSE;
}

#else /* HAVE_EGL */

/* ********************************************************************** */

#include <EGL/egl.h>
#include <EGL/eglext.h>

// Keep the core-profile request available with older EGL headers.  The
// runtime extension/version check below still decides whether the attributes
// may actually be passed to eglCreateContext().
#ifndef EGL_CONTEXT_MAJOR_VERSION_KHR
#define EGL_CONTEXT_MAJOR_VERSION_KHR 0x3098
#endif
#ifndef EGL_CONTEXT_MINOR_VERSION_KHR
#define EGL_CONTEXT_MINOR_VERSION_KHR 0x30FB
#endif
#ifndef EGL_CONTEXT_OPENGL_PROFILE_MASK_KHR
#define EGL_CONTEXT_OPENGL_PROFILE_MASK_KHR 0x30FD
#endif
#ifndef EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT_KHR
#define EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT_KHR 0x00000001
#endif

EGLDisplay eglglue_display = EGL_NO_DISPLAY;
// A display borrowed from an application's current context must not be
// terminated when Coin's EGL glue is cleaned up.
static SbBool eglglue_display_owned = FALSE;
static SbBool eglglue_display_initialized = FALSE;
static EGLint eglglue_display_major = 0;
static EGLint eglglue_display_minor = 0;
struct eglglue_contextdata;

#define CASE_STR( value ) case value: return #value;
const char* eglErrorString( EGLint error )
{
    switch( error )
    {
    CASE_STR( EGL_SUCCESS             )
    CASE_STR( EGL_NOT_INITIALIZED     )
    CASE_STR( EGL_BAD_ACCESS          )
    CASE_STR( EGL_BAD_ALLOC           )
    CASE_STR( EGL_BAD_ATTRIBUTE       )
    CASE_STR( EGL_BAD_CONTEXT         )
    CASE_STR( EGL_BAD_CONFIG          )
    CASE_STR( EGL_BAD_CURRENT_SURFACE )
    CASE_STR( EGL_BAD_DISPLAY         )
    CASE_STR( EGL_BAD_SURFACE         )
    CASE_STR( EGL_BAD_MATCH           )
    CASE_STR( EGL_BAD_PARAMETER       )
    CASE_STR( EGL_BAD_NATIVE_PIXMAP   )
    CASE_STR( EGL_BAD_NATIVE_WINDOW   )
    CASE_STR( EGL_CONTEXT_LOST        )
    default: return "Unknown";
    }
}
const char* eglAPIString( EGLenum api )
{
    switch( api )
    {
    CASE_STR( EGL_OPENGL_API          )
    CASE_STR( EGL_OPENGL_ES_API          )
    CASE_STR( EGL_OPENVG_API          )
    default: return "Unknown";
    }
}
#undef CASE_STR

struct eglglue_binding {
  EGLenum api;
  EGLDisplay display;
  EGLContext context;
  EGLSurface drawSurface;
  EGLSurface readSurface;
};

static eglglue_binding
eglglue_capture_binding(void)
{
  eglglue_binding binding;
  binding.api = eglQueryAPI();
  binding.display = eglGetCurrentDisplay();
  binding.context = eglGetCurrentContext();
  binding.drawSurface = eglGetCurrentSurface(EGL_DRAW);
  binding.readSurface = eglGetCurrentSurface(EGL_READ);
  return binding;
}

static SbBool
eglglue_bind_api(EGLenum api)
{
  return api == EGL_NONE || eglBindAPI(api) == EGL_TRUE;
}

static SbBool
eglglue_restore_binding(const eglglue_binding & binding,
                        EGLDisplay fallbackDisplay)
{
  const EGLDisplay display = binding.display != EGL_NO_DISPLAY
      ? binding.display : fallbackDisplay;
  if (display == EGL_NO_DISPLAY || !eglglue_bind_api(binding.api)) {
    return FALSE;
  }
  return eglMakeCurrent(display,
                        binding.drawSurface,
                        binding.readSurface,
                        binding.context) == EGL_TRUE;
}

struct eglglue_contextdata {
  EGLDisplay display;
  EGLContext context;
  EGLSurface surface;
  EGLConfig config;
  eglglue_binding previousBinding;
  unsigned int width;
  unsigned int height;
};

static struct eglglue_contextdata *
eglglue_contextdata_init(unsigned int width, unsigned int height)
{
  struct eglglue_contextdata * ctx;
  ctx = static_cast<struct eglglue_contextdata *>(malloc(sizeof(*ctx)));
  if (!ctx) return NULL;

  ctx->display = EGL_NO_DISPLAY;
  ctx->context = EGL_NO_CONTEXT;
  ctx->surface = EGL_NO_SURFACE;
  ctx->config = (EGLConfig) 0;
  ctx->previousBinding.api = EGL_NONE;
  ctx->previousBinding.display = EGL_NO_DISPLAY;
  ctx->previousBinding.context = EGL_NO_CONTEXT;
  ctx->previousBinding.drawSurface = EGL_NO_SURFACE;
  ctx->previousBinding.readSurface = EGL_NO_SURFACE;
  ctx->width = width;
  ctx->height = height;
  return ctx;
}

static EGLDisplay
eglglue_acquire_display(void)
{
  PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT;
  const char * platform = coin_getenv("EGL_PLATFORM");
  const EGLDisplay currentDisplay = eglGetCurrentDisplay();

  // Reuse an active EGL display so Coin follows the platform selected by the
  // application. Fall back to EGL_DEFAULT_DISPLAY only when no EGL context
  // is current.
  // Surfaceless rendering is selected explicitly because it has no native
  // Wayland or X11 display to probe.
  const bool requestedSurfaceless =
      platform && strcmp(platform, "surfaceless") == 0;

  if (!requestedSurfaceless && currentDisplay != EGL_NO_DISPLAY) {
    eglglue_display = currentDisplay;
    eglglue_display_owned = FALSE;
    return eglglue_display;
  }

  eglGetPlatformDisplayEXT =
      (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");

#ifdef EGL_PLATFORM_SURFACELESS_MESA
  if (requestedSurfaceless) {
    if (!eglGetPlatformDisplayEXT) {
      cc_debugerror_post("eglglue_acquire_display",
                         "EGL_PLATFORM=surfaceless requested, but "
                         "eglGetPlatformDisplayEXT is unavailable.");
      return EGL_NO_DISPLAY;
    }
    eglglue_display = eglGetPlatformDisplayEXT(
        EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL);
    if (eglglue_display == EGL_NO_DISPLAY) {
      cc_debugerror_post("eglglue_acquire_display",
                         "EGL_PLATFORM=surfaceless requested, but the "
                         "surfaceless display is unavailable. %s",
                         eglErrorString(eglGetError()));
      return EGL_NO_DISPLAY;
    }
  } else {
    eglglue_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  }
#else
  if (requestedSurfaceless) {
    cc_debugerror_post("eglglue_acquire_display",
                       "EGL_PLATFORM=surfaceless requested, but the "
                       "surfaceless Mesa platform is unavailable.");
    return EGL_NO_DISPLAY;
  }
  eglglue_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
#endif

  if (eglglue_display == EGL_NO_DISPLAY) {
    cc_debugerror_post("eglglue_acquire_display",
                       "Could not obtain the default EGL display. %s",
                       eglErrorString(eglGetError()));
    return EGL_NO_DISPLAY;
  }

  eglglue_display_owned = TRUE;

  if (coin_glglue_debug()) {
    cc_debugerror_postinfo("eglglue_acquire_display",
                            "got EGLDisplay==%p",
                            eglglue_display);
  }

  return eglglue_display;
}

static EGLDisplay
eglglue_get_display(void)
{
  if (eglglue_display == EGL_NO_DISPLAY) {
    eglglue_display = eglglue_acquire_display();
  }
  return eglglue_display;
}

static SbBool
eglglue_initialize_display(EGLDisplay display, EGLint * major, EGLint * minor)
{
  if (display == EGL_NO_DISPLAY) {
    return FALSE;
  }

  if (display == eglglue_display && eglglue_display_initialized) {
    if (major) *major = eglglue_display_major;
    if (minor) *minor = eglglue_display_minor;
    return TRUE;
  }

  EGLint initializedMajor;
  EGLint initializedMinor;
  if (eglInitialize(display, &initializedMajor, &initializedMinor) == EGL_FALSE) {
    return FALSE;
  }

  if (display == eglglue_display) {
    eglglue_display_major = initializedMajor;
    eglglue_display_minor = initializedMinor;
    eglglue_display_initialized = TRUE;
  }
  if (major) *major = initializedMajor;
  if (minor) *minor = initializedMinor;
  return TRUE;
}

void
eglglue_init(cc_glglue * w)
{
  EGLDisplay display = eglglue_get_display();
  const EGLenum previousApi = eglQueryAPI();
  w->glx.isdirect = 1;
  w->glx.serverversion = NULL;
  w->glx.servervendor = NULL;
  w->glx.serverextensions = NULL;
  w->glx.clientversion = NULL;
  w->glx.clientvendor = NULL;
  w->glx.clientextensions = NULL;
  w->glx.glxextensions = NULL;

  w->glx.glXGetCurrentDisplay = (COIN_PFNGLXGETCURRENTDISPLAYPROC)eglglue_getprocaddress(w, "eglglue_get_display");

  if (!eglglue_initialize_display(display,
                                  &w->glx.version.major,
                                  &w->glx.version.minor)) {
    cc_debugerror_post("eglglue_init",
                       "Couldn't initialize EGL. %s",
                        eglErrorString(eglGetError()));
    return;
  }

  if (!eglglue_bind_api(EGL_OPENGL_API)) {
    cc_debugerror_post("eglglue_init",
                       "eglBindAPI(EGL_OPENGL_API) failed. %s",
                       eglErrorString(eglGetError()));
    return;
  }

  if (coin_glglue_debug()) {
    cc_debugerror_postinfo("eglglue_init",
                           "EGL version: %d.%d",
                            w->glx.version.major,
                            w->glx.version.minor);
    cc_debugerror_postinfo("eglglue_init",
                           "eglQueryString(EGL_VERSION)=='%s'",
                            eglQueryString(eglglue_get_display(), EGL_VERSION));
    cc_debugerror_postinfo("eglglue_init",
                           "eglQueryString(EGL_VENDOR)=='%s'",
                            eglQueryString(eglglue_get_display(), EGL_VENDOR));
    cc_debugerror_postinfo("eglglue_init",
                           "eglQueryString(EGL_CLIENT_APIS)=='%s'",
                            eglQueryString(eglglue_get_display(), EGL_CLIENT_APIS));
    cc_debugerror_postinfo("eglglue_init",
                           "eglQueryAPI()=='%s'",
                           eglAPIString(eglQueryAPI()));
    cc_debugerror_postinfo("eglglue_init",
                           "eglQueryString(EGL_EXTENSIONS)=='%s'",
                            eglQueryString(eglglue_get_display(), EGL_EXTENSIONS));
  }

  if (previousApi != EGL_NONE && !eglglue_bind_api(previousApi)) {
    cc_debugerror_post("eglglue_init",
                       "Could not restore the EGL client API (%s). %s",
                       eglAPIString(previousApi),
                       eglErrorString(eglGetError()));
  }
}

static void
eglglue_contextdata_cleanup(struct eglglue_contextdata * ctx)
{
  if (ctx == NULL) { return; }
  EGLDisplay display = ctx->display;
  if (display != EGL_NO_DISPLAY && ctx->context != EGL_NO_CONTEXT) {
    if (eglGetCurrentContext() == ctx->context) {
      eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    }
    eglDestroyContext(display, ctx->context);
  }
  if (display != EGL_NO_DISPLAY && ctx->surface != EGL_NO_SURFACE) {
    eglDestroySurface(display, ctx->surface);
  }
  free(ctx);
}

static SbBool
eglglue_core_profile_requested(void)
{
  const char * coreprofile = coin_getenv("COIN_EGL_CORE_PROFILE");
  return coreprofile && atoi(coreprofile) > 0;
}

static SbBool
eglglue_core_profile_supported(EGLDisplay display, EGLint major, EGLint minor)
{
  const char * extensions = eglQueryString(display, EGL_EXTENSIONS);
  return (major > 1 || (major == 1 && minor >= 5)) ||
      (extensions != NULL &&
       coin_glglue_extension_available(extensions, "EGL_KHR_create_context"));
}

static void
eglglue_context_attributes(SbBool requestCoreProfile, EGLint * attributes)
{
  if (!requestCoreProfile) {
    attributes[0] = EGL_NONE;
    return;
  }

  attributes[0] = EGL_CONTEXT_MAJOR_VERSION_KHR;
  attributes[1] = 3;
  attributes[2] = EGL_CONTEXT_MINOR_VERSION_KHR;
  attributes[3] = 3;
  attributes[4] = EGL_CONTEXT_OPENGL_PROFILE_MASK_KHR;
  attributes[5] = EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT_KHR;
  attributes[6] = EGL_NONE;
}

void *
eglglue_context_create_offscreen(unsigned int width, unsigned int height)
{
  struct eglglue_contextdata * ctx = NULL;
  EGLint numConfigs = 0;
  EGLDisplay display = EGL_NO_DISPLAY;
  EGLint eglmajor = 0;
  EGLint eglminor = 0;
  EGLenum previousApi = EGL_NONE;
  SbBool requestCoreProfile = FALSE;
  SbBool success = FALSE;
  EGLint attrib[] = {
    EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
    EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
    EGL_RED_SIZE, 8,
    EGL_GREEN_SIZE, 8,
    EGL_BLUE_SIZE, 8,
    EGL_ALPHA_SIZE, 8,
    EGL_DEPTH_SIZE, 24,
    EGL_STENCIL_SIZE, 1,
    EGL_NONE
  };
  EGLint surface_attrib[] = {
    EGL_WIDTH, (EGLint) width,
    EGL_HEIGHT, (EGLint) height,
    EGL_NONE
  };
  EGLint context_attribs[7];

  ctx = eglglue_contextdata_init(width, height);
  if (!ctx) return NULL;

  const int v = coin_glglue_stencil_bits_hack();
  if (v != -1) {
    attrib[15] = v;
  }

  if (coin_glglue_debug()) {
    cc_debugerror_postinfo("eglglue_context_create_offscreen",
                           "Creating offscreen context.");
  }

  previousApi = eglQueryAPI();
  display = eglglue_get_display();
  ctx->display = display;
  if (!eglglue_initialize_display(display, &eglmajor, &eglminor)) {
    cc_debugerror_post("eglglue_context_create_offscreen",
                       "eglInitialize failed. %s",
                       eglErrorString(eglGetError()));
    goto cleanup;
  }

  if (!eglglue_bind_api(EGL_OPENGL_API)) {
    cc_debugerror_post("eglglue_context_create_offscreen",
                       "eglBindAPI(EGL_OPENGL_API) failed. %s",
                       eglErrorString(eglGetError()));
    goto cleanup;
  }

  requestCoreProfile = eglglue_core_profile_requested();
  if (requestCoreProfile) {
    if (!eglglue_core_profile_supported(display, eglmajor, eglminor)) {
      cc_debugerror_post("eglglue_context_create_offscreen",
                         "COIN_EGL_CORE_PROFILE requested, but EGL does not "
                         "support core-profile context attributes.");
      goto cleanup;
    }
  }

  if (eglChooseConfig(display, attrib, &ctx->config, 1, &numConfigs) == EGL_FALSE) {
    cc_debugerror_post("eglglue_context_create_offscreen",
                       "eglChooseConfig failed. %s",
                       eglErrorString(eglGetError()));
    goto cleanup;
  }
  if (numConfigs == 0) {
    cc_debugerror_post("eglglue_context_create_offscreen",
                       "No matching EGL pbuffer config. %s",
                       eglErrorString(eglGetError()));
    goto cleanup;
  }

  ctx->surface = eglCreatePbufferSurface(display, ctx->config, surface_attrib);

  if (ctx->surface == EGL_NO_SURFACE) {
    cc_debugerror_post("eglglue_context_create_offscreen",
                       "Couldn't create EGL surface. %s",
                       eglErrorString(eglGetError()));
    goto cleanup;
  }

  eglglue_context_attributes(requestCoreProfile, context_attribs);
  ctx->context = eglCreateContext(display, ctx->config, EGL_NO_CONTEXT,
                                  context_attribs);

  if (ctx->context == EGL_NO_CONTEXT) {
    cc_debugerror_post("eglglue_context_create_offscreen",
                       "Couldn't create EGL context. %s",
                       eglErrorString(eglGetError()));
    goto cleanup;
  }

  success = TRUE;
  if (coin_glglue_debug()) {
    cc_debugerror_postinfo("eglglue_context_create_offscreen",
                           "created new pBuffer offscreen context == %p",
                           ctx->context);
  }

cleanup:
  if (previousApi != EGL_NONE && !eglglue_bind_api(previousApi)) {
    cc_debugerror_post("eglglue_context_create_offscreen",
                       "Could not restore the EGL client API (%s). %s",
                       eglAPIString(previousApi),
                       eglErrorString(eglGetError()));
    success = FALSE;
  }
  if (!success) {
    eglglue_contextdata_cleanup(ctx);
    ctx = NULL;
  }
  return ctx;
}

SbBool
eglglue_context_make_current(void * ctx)
{
  struct eglglue_contextdata * context = (struct eglglue_contextdata *)ctx;
  if (context == NULL || context->display == EGL_NO_DISPLAY) {
    return FALSE;
  }

  context->previousBinding = eglglue_capture_binding();
  if (!eglglue_bind_api(EGL_OPENGL_API)) {
    cc_debugerror_post("eglglue_context_make_current",
                       "eglBindAPI(EGL_OPENGL_API) failed: %s",
                       eglErrorString(eglGetError()));
    return FALSE;
  }

  if (eglMakeCurrent(context->display,
                     context->surface,
                     context->surface,
                     context->context) == EGL_FALSE) {
      cc_debugerror_post("eglglue_context_make_current",
                         "eglMakeCurrent failed: %s",
                         eglErrorString(eglGetError()));
      if (!eglglue_restore_binding(context->previousBinding, context->display)) {
        cc_debugerror_post("eglglue_context_make_current",
                           "Could not restore the caller EGL binding: %s",
                           eglErrorString(eglGetError()));
      }
      return FALSE;
  }

  if (coin_glglue_debug()) {
      cc_debugerror_postinfo("eglglue_context_make_current",
                             "EGL Context (0x%X)\n",
                             context->context);
  }
  return TRUE;
}

void
eglglue_context_reinstate_previous(void * ctx)
{
  struct eglglue_contextdata * context = (struct eglglue_contextdata *)ctx;
  if (context == NULL || context->display == EGL_NO_DISPLAY) {
    return;
  }

  // The API must be restored before the context and its surfaces are made
  // current. The display stored with the caller binding is required when it
  // belongs to a foreign EGL display; the Coin display is only the fallback
  // for restoring EGL_NO_CONTEXT.
  if (!eglglue_restore_binding(context->previousBinding, context->display)) {
    cc_debugerror_post("eglglue_context_reinstate_previous",
                       "Could not restore the caller EGL binding: %s",
                       eglErrorString(eglGetError()));
  } else if (coin_glglue_debug()) {
    cc_debugerror_postinfo("eglglue_context_reinstate_previous",
                           "restored caller EGL binding");
  }
}

void
eglglue_context_destruct(void * ctx)
{
  struct eglglue_contextdata * context = (struct eglglue_contextdata *)ctx;

  if (coin_glglue_debug()) {
    cc_debugerror_postinfo("eglglue_context_destruct",
                           "Destroying context %p", context->context);
  }
  eglglue_contextdata_cleanup(context);
}

void
eglglue_context_bind_pbuffer(void * ctx)
{
  struct eglglue_contextdata * context = (struct eglglue_contextdata *)ctx;

  if (eglBindTexImage(context->display, context->surface, EGL_BACK_BUFFER) == EGL_FALSE) {
    cc_debugerror_post("eglglue_context_bind_pbuffer()"
                       "after binding pbuffer: %s",
                       eglErrorString(eglGetError()));
  }
}

void
eglglue_context_release_pbuffer(void * ctx)
{
  struct eglglue_contextdata * context = (struct eglglue_contextdata *)ctx;

  if (eglReleaseTexImage(context->display, context->surface, EGL_BACK_BUFFER) == EGL_FALSE) {
    cc_debugerror_post("eglglue_context_release_pbuffer()"
                       "releasing pbuffer: %s",
                       eglErrorString(eglGetError()));
  }
}

SbBool
eglglue_context_pbuffer_is_bound(void * ctx)
{
  struct eglglue_contextdata * context = (struct eglglue_contextdata *)ctx;
  GLint buffer = EGL_NONE;

  if(eglQueryContext(context->display, context->context, EGL_RENDER_BUFFER, &buffer) == EGL_FALSE) {
    cc_debugerror_post("eglglue_context_pbuffer_is_bound()"
                       "after query pbuffer: %s",
                       eglErrorString(eglGetError()));
  }
  return buffer == EGL_BACK_BUFFER;
}

SbBool
eglglue_context_can_render_to_texture(void * COIN_UNUSED_ARG(ctx))
{
  /* Surfaceless pbuffers are not EGL texture targets in this implementation. */
  return FALSE;
}

SbBool
eglglue_context_pbuffer_max(void * ctx, unsigned int * lims)
{
  int returnval, attribval, i;
  const int attribs[] = {
    EGL_MAX_PBUFFER_WIDTH, EGL_MAX_PBUFFER_HEIGHT, EGL_MAX_PBUFFER_PIXELS
  };
  struct eglglue_contextdata * context = (struct eglglue_contextdata *)ctx;

  if (context->surface == EGL_NO_SURFACE) { return FALSE; }

  for (i = 0; i < 3; i++) {
    if(eglGetConfigAttrib(context->display, context->config, attribs[i], &attribval) == EGL_FALSE) {
      cc_debugerror_post("eglglue_context_pbuffer_max",
                         "eglGetConfigAttrib() failed, "
                         "returned error code %s",
                         eglErrorString(eglGetError()));
      return FALSE;
    }
    assert(attribval >= 0);
    // EGL permits EGL_MAX_PBUFFER_PIXELS to be reported as zero when the
    // implementation does not impose a pixel-count limit. Coin uses zero as
    // an actual limit, so normalize that value to the representable maximum.
    lims[i] = (attribs[i] == EGL_MAX_PBUFFER_PIXELS && attribval == 0)
        ? UINT_MAX
        : (unsigned int)attribval;
  }
  return TRUE;
}

void *
eglglue_getprocaddress(const cc_glglue * glue_in, const char * fname)
{
  return (void *)eglGetProcAddress(fname);
}

void
eglglue_cleanup(void)
{
  if (eglglue_display_owned && eglglue_display != EGL_NO_DISPLAY) {
    eglTerminate(eglglue_display);
  }
  eglglue_display = EGL_NO_DISPLAY;
  eglglue_display_owned = FALSE;
  eglglue_display_initialized = FALSE;
  eglglue_display_major = 0;
  eglglue_display_minor = 0;
}

#endif /* HAVE_EGL */
