#ifndef COIN_SOSCENEMANAGERBASE_H
#define COIN_SOSCENEMANAGERBASE_H

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

// Internal, backend-agnostic owner of the state that the legacy OpenGL
// SoRenderManager and the Vulkan SoVulkanRenderManager both need: the scene
// graph root, the camera, the viewport region and device-pixel ratio, the
// clear/background color, the active flag, and the redraw callback/scheduling
// machinery.  It exists so the two managers no longer hand-mirror that state
// and its ref-counting; each renderer keeps only its backend-specific state
// (GL stereo/superimpositions/GL actions, Vulkan overlays/backends/replay).
//
// The class is compiled into libCoin unconditionally; it is deliberately not
// installed, so it is not part of the public ABI.

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif // HAVE_CONFIG_H

#include <stdint.h>

#include <Inventor/SbBasic.h>
#include <Inventor/SbColor4f.h>
#include <Inventor/SbVec2s.h>
#include <Inventor/SbViewportRegion.h>

class SoNode;
class SoCamera;
class SoOneShotSensor;
class SoSensor;

// The render callback is stored type-erased: the second argument is the
// embedding manager object (whatever concrete manager owns this base), so the
// base itself stays free of any renderer's type.  The legacy SoRenderManager
// hands out SoRenderManagerRenderCB (void(*)(void*, SoRenderManager*)); the
// conversion happens in its own translation unit.
typedef void SoSceneManagerBaseRenderCB(void * userdata, void * manager);

class SoSceneManagerBase {
public:
  SoSceneManagerBase(uint32_t defaultRedrawPriority = 10000,
                     void * publicManager = NULL);
  virtual ~SoSceneManagerBase();

  // --- scene graph ------------------------------------------------------
  void setSceneGraph(SoNode * sceneroot);
  SoNode * getSceneGraph(void) const { return this->scene; }

  // --- camera -----------------------------------------------------------
  void setCamera(SoCamera * camera);
  SoCamera * getCamera(void) const { return this->camera; }

  // --- viewport region + device-pixel ratio -----------------------------
  void setViewportRegion(const SbViewportRegion & region)
  { this->viewport = region; }
  const SbViewportRegion & getViewportRegion(void) const
  { return this->viewport; }
  void setWindowSize(const SbVec2s & newsize);
  const SbVec2s & getWindowSize(void) const
  { return this->viewport.getWindowSize(); }
  void setSize(const SbVec2s & newsize);
  const SbVec2s & getSize(void) const
  { return this->viewport.getViewportSizePixels(); }
  void setOrigin(const SbVec2s & newOrigin);
  const SbVec2s & getOrigin(void) const
  { return this->viewport.getViewportOriginPixels(); }
  void setDevicePixelRatio(float dpr) { this->devicePixelRatio = dpr; }
  float getDevicePixelRatio(void) const { return this->devicePixelRatio; }

  // --- background -------------------------------------------------------
  void setBackgroundColor(const SbColor4f & color)
  { this->backgroundcolor = color; }
  const SbColor4f & getBackgroundColor(void) const
  { return this->backgroundcolor; }

  // --- activation + redraw ---------------------------------------------
  void activate(void) { this->isactive = TRUE; }
  void deactivate(void) { this->isactive = FALSE; }
  int isActive(void) const { return this->isactive; }

  void setRenderCallback(SoSceneManagerBaseRenderCB * f,
                         void * userdata = NULL);
  SbBool isAutoRedraw(void) const { return this->rendercb != NULL; }
  void scheduleRedraw(void);
  void setRedrawPriority(uint32_t priority);
  uint32_t getRedrawPriority(void) const { return this->redrawpri; }
  void redraw(void);

  // Shared state.  Public (rather than accessor-only) so the existing
  // implementation translation units of both managers, which reach it through
  // their pimpl pointer, keep compiling unchanged.
  SoNode * scene;
  SoCamera * camera;
  SbViewportRegion viewport;
  float devicePixelRatio;
  SbColor4f backgroundcolor;
  SbBool isactive;
  SoSceneManagerBaseRenderCB * rendercb;
  void * rendercbdata;
  uint32_t redrawpri;
  SoOneShotSensor * redrawshot;
  // The public manager object handed to the render callback at redraw time.
  void * publicManager;

protected:
  // Renderer-specific reaction to a new scene-graph root (GL attaches its root
  // / clipping sensors, Vulkan re-arms its graph-dirty sensor and drops its
  // bounding-box cache).  Called with `newroot` already retained and before the
  // previous root is released.
  virtual void sceneGraphChanged(SoNode * oldroot, SoNode * newroot);
  static void redrawSensorCB(void * data, SoSensor * sensor);

private:
  SoSceneManagerBase(const SoSceneManagerBase & rhs);
  SoSceneManagerBase & operator=(const SoSceneManagerBase & rhs);
};

#endif // !COIN_SOSCENEMANAGERBASE_H
