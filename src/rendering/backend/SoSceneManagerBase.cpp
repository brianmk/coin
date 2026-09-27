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

#include "SoSceneManagerBase.h"

#include <Inventor/nodes/SoNode.h>
#include <Inventor/nodes/SoCamera.h>
#include <Inventor/sensors/SoOneShotSensor.h>
#include <Inventor/sensors/SoSensor.h>

// The ref-counting, viewport arithmetic and background/camera state below are
// moved verbatim from SoRenderManagerP; keeping the exact same operations is
// what lets the legacy OpenGL manager delegate to the base without changing
// its behaviour.

SoSceneManagerBase::SoSceneManagerBase(uint32_t defaultRedrawPriority,
                                       void * publicMgr)
  : scene(NULL),
    camera(NULL),
    viewport(SbVec2s(1, 1)),
    devicePixelRatio(1.0f),
    backgroundcolor(0.0f, 0.0f, 0.0f, 1.0f),
    isactive(FALSE),
    rendercb(NULL),
    rendercbdata(NULL),
    redrawpri(defaultRedrawPriority),
    redrawshot(NULL),
    publicManager(publicMgr)
{
  this->redrawshot =
    new SoOneShotSensor(SoSceneManagerBase::redrawSensorCB, this);
  this->redrawshot->setPriority(this->redrawpri);
}

SoSceneManagerBase::~SoSceneManagerBase()
{
  if (this->scene) this->scene->unref();
  if (this->camera) this->camera->unref();
  delete this->redrawshot;
}

// Don't unref() the old root until after the new one is set up, in case the
// old root == the new sceneroot.
void
SoSceneManagerBase::setSceneGraph(SoNode * sceneroot)
{
  SoNode * oldroot = this->scene;

  this->scene = sceneroot;
  if (this->scene) this->scene->ref();

  this->sceneGraphChanged(oldroot, sceneroot);

  if (oldroot) oldroot->unref();
}

void
SoSceneManagerBase::setCamera(SoCamera * cam)
{
  // avoid unref() then ref() on the same node
  if (cam == this->camera) return;

  if (this->camera) this->camera->unref();
  this->camera = cam;
  if (cam) cam->ref();
}

void
SoSceneManagerBase::setWindowSize(const SbVec2s & newsize)
{
  SbViewportRegion region = this->viewport;
  region.setWindowSize(newsize[0], newsize[1]);
  this->viewport = region;
}

void
SoSceneManagerBase::setSize(const SbVec2s & newsize)
{
  SbViewportRegion region = this->viewport;
  SbVec2s origin = region.getViewportOriginPixels();
  region.setViewportPixels(origin, newsize);
  this->viewport = region;
}

void
SoSceneManagerBase::setOrigin(const SbVec2s & newOrigin)
{
  SbViewportRegion region = this->viewport;
  SbVec2s size = region.getViewportSizePixels();
  region.setViewportPixels(newOrigin, size);
  this->viewport = region;
}

void
SoSceneManagerBase::setRenderCallback(SoSceneManagerBaseRenderCB * f,
                                      void * userdata)
{
  this->rendercb = f;
  this->rendercbdata = userdata;
}

void
SoSceneManagerBase::scheduleRedraw(void)
{
  if (this->isactive && this->rendercb && this->redrawshot) {
    this->redrawshot->schedule();
  }
}

void
SoSceneManagerBase::setRedrawPriority(uint32_t priority)
{
  this->redrawpri = priority;
  if (this->redrawshot) this->redrawshot->setPriority(priority);
}

void
SoSceneManagerBase::redraw(void)
{
  if (this->rendercb) {
    this->rendercb(this->rendercbdata, this->publicManager);
  }
}

void
SoSceneManagerBase::redrawSensorCB(void * data, SoSensor * /* sensor */)
{
  SoSceneManagerBase * self = static_cast<SoSceneManagerBase *>(data);

  // Need to recheck the "active" flag, as it could have changed since it was
  // tested in scheduleRedraw().
  if (self->isactive) {
    self->redraw();
  }
}

void
SoSceneManagerBase::sceneGraphChanged(SoNode * /* oldroot */,
                                      SoNode * /* newroot */)
{
  // Default: no renderer-specific bookkeeping.
}
