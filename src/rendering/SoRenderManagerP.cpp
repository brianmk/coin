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

#include "SoRenderManagerP.h"
#include "coindefs.h"
#include "SoClippingPlanes.h"

#include <limits>

#include <Inventor/nodes/SoInfo.h>
#include <Inventor/nodes/SoCamera.h>
#include <Inventor/nodes/SoPerspectiveCamera.h>
#include "Inventor/nodes/SoOrthographicCamera.h"
#if COIN_BUILD_LEGACY_GL_RENDERER
#include <Inventor/actions/SoGLRenderAction.h>
#endif
#include <Inventor/actions/SoGetBoundingBoxAction.h>
#include <Inventor/actions/SoGetMatrixAction.h>
#include <Inventor/actions/SoSearchAction.h>

SbBool SoRenderManagerP::touchtimer = TRUE;
SbBool SoRenderManagerP::cleanupfunctionset = FALSE;
int SoRenderManagerRootSensor::debugrootnotifications = -1;

#define PRIVATE(p) (p->pimpl)
#define PUBLIC(p) (p->publ)

#define INHERIT_TRANSPARENCY_TYPE -1

SoRenderManagerP::SoRenderManagerP(SoRenderManager * publ)
{
  this->publ = publ;
  this->getmatrixaction = NULL;
  this->getbboxaction = NULL;
  this->searchaction = NULL;
}

SoRenderManagerP::~SoRenderManagerP()
{
  delete this->getmatrixaction;
  delete this->getbboxaction;
  delete this->searchaction;
}

// Internal callback.
void
SoRenderManagerP::redrawshotTriggeredCB(void * data, SoSensor * /* sensor */)
{
#if COIN_DEBUG && 0 // debug
  SoDebugError::postInfo("SoRenderManager::redrawshotTriggeredCB", "start");
#endif // debug

  SoRenderManager * thisp = (SoRenderManager *) data;

  // Need to recheck the "active" flag, as it could have changed since
  // it was tested in the SoRenderManager::scheduleRedraw() call.
  if (PRIVATE(thisp)->isactive) { thisp->redraw(); }

#if COIN_DEBUG && 0 // debug
  SoDebugError::postInfo("SoRenderManager::redrawshotTriggeredCB", "done\n\n");
#endif // debug
}

void
SoRenderManagerP::cleanup(void)
{
  SoRenderManagerP::touchtimer = TRUE;
  SoRenderManagerP::cleanupfunctionset = FALSE;
}

void
SoRenderManagerP::updateClippingPlanesCB(void * COIN_UNUSED_ARG(closure), SoSensor * COIN_UNUSED_ARG(sensor))
{
  //SoRenderManagerP * thisp = (SoRenderManagerP *) closure;
  //if (thisp->autoclipping != SoRenderManager::NO_AUTO_CLIPPING) {
  //  thisp->setClippingPlanes();
  //}
}

void
SoRenderManagerP::setClippingPlanes(void)
{
#if !COIN_BUILD_LEGACY_GL_RENDERER
  return;
#else
  SoCamera * camera = this->camera;
  SoNode * scene = this->scene;
  if (!camera || !scene) return;

  SbViewportRegion vp = this->viewport;

  if (!this->getbboxaction) {
    this->getbboxaction = new SoGetBoundingBoxAction(vp);
  } else {
    this->getbboxaction->setViewportRegion(vp);
  }
  this->getbboxaction->apply(scene);

  SbXfBox3f xbox = this->getbboxaction->getXfBoundingBox();
  SbMatrix cammat;
  SbMatrix inverse;
  this->getCameraCoordinateSystem(cammat, inverse);
  xbox.transform(inverse);

  SbMatrix mat;
  mat.setTranslate(- camera->position.getValue());
  xbox.transform(mat);
  mat = camera->orientation.getValue().inverse();
  xbox.transform(mat);
  SbBox3f box = xbox.project();

  // Shared near/far computation (diagonal offset, empty-box defaults,
  // perspective near limit) with the Vulkan manager.
  float nearval, farval;
  if (!coinComputeClippingPlanes(
        box,
        camera->isOfType(SoOrthographicCamera::getClassTypeId()) ? true : false,
        camera->isOfType(SoPerspectiveCamera::getClassTypeId()) ? true : false,
        static_cast<int>(this->autoclipping),
        this->nearplanevalue,
        nearval,
        farval)) {
    return;
  }

  const float SLACK = kSoClippingSlack;
  const float newnear = nearval >= 0 ? nearval * (1.0f - SLACK) : nearval * (1.0f + SLACK);
  const float newfar = farval >= 0 ? farval * (1.0f + SLACK) : farval * (1.0f - SLACK);

  const float neareps = nearval * SLACK * SLACK;
  const float fareps = farval * SLACK * SLACK;

  const float oldnear = camera->nearDistance.getValue();
  const float oldfar = camera->farDistance.getValue();

  // check that the values have changed before setting the fields to
  // avoid continuous redraws on static scenes. Use an epsilon value
  // when comparing

  if (SbAbs(oldnear - newnear) > SbAbs(neareps)) {
    camera->nearDistance = newnear;
  }
  if (SbAbs(oldfar - newfar) > SbAbs(fareps)) {
    camera->farDistance = newfar;
  }
#endif
}

void
SoRenderManagerP::getCameraCoordinateSystem(SbMatrix & matrix,
                                            SbMatrix & inverse)
{
#if !COIN_BUILD_LEGACY_GL_RENDERER
  matrix = inverse = SbMatrix::identity();
  return;
#else
  SoCamera * camera = this->camera;
  SoNode * scene = this->scene;
  assert(camera && scene);

  matrix = inverse = SbMatrix::identity();

  if (!this->searchaction) {
    this->searchaction = new SoSearchAction;
  }

  this->searchaction->reset();
  this->searchaction->setSearchingAll(TRUE);
  this->searchaction->setInterest(SoSearchAction::FIRST);
  this->searchaction->setNode(camera);
  this->searchaction->apply(scene);

  if (this->searchaction->getPath()) {
    if (!this->getmatrixaction) {
      this->getmatrixaction =
        new SoGetMatrixAction(this->viewport);
    } else {
      this->getmatrixaction->setViewportRegion(this->viewport);
    }
    this->getmatrixaction->apply(this->searchaction->getPath());
    matrix = this->getmatrixaction->getMatrix();
    inverse = this->getmatrixaction->getInverse();
  }
  this->searchaction->reset();
#endif
}

//**********************************************************************************
// Superimposition
//**********************************************************************************

class SuperimpositionP {
public:
  SoNode * scene;
  SbBool enabled;
  SoRenderManager * manager;
  SoNodeSensor * sensor;
  uint32_t stateflags;
  int transparencytype;
};

SoRenderManager::Superimposition::Superimposition(SoNode * scene,
                                                  SbBool enabled,
                                                  SoRenderManager * manager,
                                                  uint32_t flags)
{
  assert(scene != NULL);
  PRIVATE(this) = new SuperimpositionP;

  PRIVATE(this)->scene = scene;
  PRIVATE(this)->scene->ref();

  PRIVATE(this)->enabled = enabled;
  PRIVATE(this)->stateflags = flags;

  PRIVATE(this)->transparencytype = INHERIT_TRANSPARENCY_TYPE;

  PRIVATE(this)->manager = manager;
  PRIVATE(this)->sensor = new SoNodeSensor(Superimposition::changeCB, this);
  PRIVATE(this)->sensor->attach(PRIVATE(this)->scene);
}

SoRenderManager::Superimposition::~Superimposition()
{
  PRIVATE(this)->scene->unref();
  delete PRIVATE(this)->sensor;
  delete PRIVATE(this);
}

int
SoRenderManager::Superimposition::getStateFlags(void) const
{
  return PRIVATE(this)->stateflags;
}

#if COIN_BUILD_LEGACY_GL_RENDERER
void
SoRenderManager::Superimposition::render(SoGLRenderAction * action, SbBool clearcolorbuffer)
{
  if (!PRIVATE(this)->enabled) return;

  SoGLRenderAction::TransparencyType oldttype = action->getTransparencyType();
  if (PRIVATE(this)->transparencytype != INHERIT_TRANSPARENCY_TYPE) {
    action->setTransparencyType((SoGLRenderAction::TransparencyType) PRIVATE(this)->transparencytype);
  }
  SbBool zbufferwason = glIsEnabled(GL_DEPTH_TEST) ? TRUE : FALSE;

  PRIVATE(this)->stateflags & Superimposition::ZBUFFERON ?
    glEnable(GL_DEPTH_TEST):
    glDisable(GL_DEPTH_TEST);

  GLbitfield clearflags = clearcolorbuffer ? GL_COLOR_BUFFER_BIT : 0;
  if (PRIVATE(this)->stateflags & Superimposition::CLEARZBUFFER) {
    clearflags |= GL_DEPTH_BUFFER_BIT;
  }

  PRIVATE(this)->manager->renderScene(action, PRIVATE(this)->scene, (uint32_t) clearflags);

  zbufferwason ?
    glEnable(GL_DEPTH_TEST):
    glDisable(GL_DEPTH_TEST);

  if (PRIVATE(this)->transparencytype != INHERIT_TRANSPARENCY_TYPE) {
    action->setTransparencyType(oldttype);
  }
}
#endif

void
SoRenderManager::Superimposition::setEnabled(SbBool yes)
{
  PRIVATE(this)->enabled = yes;
}

void
SoRenderManager::Superimposition::changeCB(void * data, SoSensor * COIN_UNUSED_ARG(sensor))
{
  Superimposition * thisp = (Superimposition *) data;
  assert(thisp && PRIVATE(thisp)->manager);
  if (PRIVATE(thisp)->stateflags & Superimposition::AUTOREDRAW) {
    PRIVATE(thisp)->manager->scheduleRedraw();
  }
}

#if COIN_BUILD_LEGACY_GL_RENDERER
void
SoRenderManager::Superimposition::setTransparencyType(SoGLRenderAction::TransparencyType type)
{
  PRIVATE(this)->transparencytype = (int) type;
}
#endif

void
SoRenderManagerP::invokePreRenderCallbacks(void)
{
  std::vector<RenderCBTouple>::const_iterator cbit =
    this->preRenderCallbacks.begin();
  while (cbit != this->preRenderCallbacks.end()) {
    cbit->first(cbit->second, PUBLIC(this));
    ++cbit;
  }
}

void
SoRenderManagerP::invokePostRenderCallbacks(void)
{
  std::vector<RenderCBTouple>::const_iterator cbit =
    this->postRenderCallbacks.begin();
  while (cbit != this->postRenderCallbacks.end()) {
    cbit->first(cbit->second, PUBLIC(this));
    ++cbit;
  }
}

#undef INHERIT_TRANSPARENCY_TYPE
#undef PRIVATE
#undef PUBLIC

void
SoRenderManagerRootSensor::notify(SoNotList * l)
{
  l->print();
  (void)fprintf(stdout, "end\n");

  inherited::notify(l);
}

SbBool
SoRenderManagerRootSensor::debug(void)
{
  if (SoRenderManagerRootSensor::debugrootnotifications == -1) {
    const char * env = coin_getenv("COIN_DEBUG_ROOT_NOTIFICATIONS");
    SoRenderManagerRootSensor::debugrootnotifications = env && (atoi(env) > 0);
  }
  return SoRenderManagerRootSensor::debugrootnotifications ? TRUE : FALSE;
}
