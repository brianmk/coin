// src/rendering/SoVulkanRenderManager.cpp

#include <Inventor/rendering/SoVulkanRenderManager.h>

#include <Inventor/SbViewportRegion.h>
#include <Inventor/actions/SoIRRenderAction.h>
#include <Inventor/errors/SoDebugError.h>
#include <Inventor/nodes/SoCamera.h>
#include <Inventor/nodes/SoNode.h>
#include <Inventor/rendering/SoRenderIR.h>
#include <Inventor/rendering/SoVulkanRenderTarget.h>

#include "rendering/SoRenderBackend.h"
#include "rendering/SoVulkanRenderBackend.h"

#include <memory>

class SoVulkanRenderManagerP {
public:
  SoVulkanRenderManagerP()
    : irAction(SbViewportRegion())
  {
    this->viewportRegion.setWindowSize(1, 1);
  }

  SoNode * scene = nullptr;
  SoCamera * camera = nullptr;
  SbViewportRegion viewportRegion;
  SbColor4f backgroundColor = SbColor4f(0.0f, 0.0f, 0.0f, 1.0f);
  SbBool clearWindow = TRUE;
  SbBool clearDepth = TRUE;
  void * renderTarget = nullptr;

  SoIRRenderAction irAction;
  SoVulkanRenderBackend backend;
  SbBool backendInitialized = FALSE;

  // Traverse the scene and harvest view/projection into params.  Returns
  // FALSE when the backend or target is unavailable.
  SbBool prepareRenderParams(SbBool clearwindow,
                             SbBool clearzbuffer,
                             SoDrawList *& drawlist,
                             SoRenderParams & params);
};

SoVulkanRenderManager::SoVulkanRenderManager()
  : pimpl(new SoVulkanRenderManagerP)
{
}

SoVulkanRenderManager::~SoVulkanRenderManager()
{
  if (this->pimpl->backendInitialized) {
    this->pimpl->backend.shutdown();
  }
  delete this->pimpl;
}

void
SoVulkanRenderManager::setSceneGraph(SoNode * root)
{
  this->pimpl->scene = root;
}

SoNode *
SoVulkanRenderManager::getSceneGraph(void) const
{
  return this->pimpl->scene;
}

void
SoVulkanRenderManager::setCamera(SoCamera * camera)
{
  this->pimpl->camera = camera;
}

SoCamera *
SoVulkanRenderManager::getCamera(void) const
{
  return this->pimpl->camera;
}

void
SoVulkanRenderManager::setViewportRegion(const SbViewportRegion & region)
{
  this->pimpl->viewportRegion = region;
}

const SbViewportRegion &
SoVulkanRenderManager::getViewportRegion(void) const
{
  return this->pimpl->viewportRegion;
}

void
SoVulkanRenderManager::setBackgroundColor(const SbColor4f & color)
{
  this->pimpl->backgroundColor = color;
}

const SbColor4f &
SoVulkanRenderManager::getBackgroundColor(void) const
{
  return this->pimpl->backgroundColor;
}

void
SoVulkanRenderManager::setClearEnabled(SbBool clearwindow, SbBool clearzbuffer)
{
  this->pimpl->clearWindow = clearwindow;
  this->pimpl->clearDepth = clearzbuffer;
}

void
SoVulkanRenderManager::getClearEnabled(SbBool & clearwindow,
                                       SbBool & clearzbuffer) const
{
  clearwindow = this->pimpl->clearWindow;
  clearzbuffer = this->pimpl->clearDepth;
}

SbBool
SoVulkanRenderManager::initialize(SoVulkanDeviceContext * context)
{
  SoRenderBackendInitParams params;
  params.userData = context;
  if (!this->pimpl->backend.initialize(params)) {
    SoDebugError::postWarning("SoVulkanRenderManager::initialize",
                              "backend initialization failed");
    return FALSE;
  }
  this->pimpl->backendInitialized = TRUE;
  return TRUE;
}

void
SoVulkanRenderManager::shutdown(void)
{
  if (this->pimpl->backendInitialized) {
    this->pimpl->backend.shutdown();
    this->pimpl->backendInitialized = FALSE;
  }
}

void
SoVulkanRenderManager::setRenderTarget(void * target)
{
  this->pimpl->renderTarget = target;
}

void *
SoVulkanRenderManager::getRenderTarget(void) const
{
  return this->pimpl->renderTarget;
}

SbBool
SoVulkanRenderManager::render(SbBool clearwindow, SbBool clearzbuffer)
{
  SoRenderParams params;
  SoDrawList * drawlist = nullptr;
  if (!this->pimpl->prepareRenderParams(clearwindow, clearzbuffer, drawlist,
                                        params)) {
    return FALSE;
  }
  if (!this->pimpl->backend.render(*drawlist, params)) {
    SoDebugError::postWarning("SoVulkanRenderManager::render",
                              "backend render failed (%d draw commands)",
                              drawlist->getNumCommands());
    return FALSE;
  }
  return TRUE;
}

SbBool
SoVulkanRenderManager::renderExternal(SbBool clearwindow,
                                      SbBool clearzbuffer,
                                      VkCommandBuffer commandBuffer,
                                      VkRenderPass renderPass)
{
  SoRenderParams params;
  SoDrawList * drawlist = nullptr;
  if (!this->pimpl->prepareRenderParams(clearwindow, clearzbuffer, drawlist,
                                        params)) {
    return FALSE;
  }
  if (!this->pimpl->backend.renderExternal(*drawlist, params, commandBuffer,
                                           renderPass)) {
    SoDebugError::postWarning("SoVulkanRenderManager::renderExternal",
                              "backend render failed (%d draw commands)",
                              drawlist->getNumCommands());
    return FALSE;
  }
  return TRUE;
}

SbBool
SoVulkanRenderManagerP::prepareRenderParams(SbBool clearwindow,
                                            SbBool clearzbuffer,
                                            SoDrawList *& drawlist,
                                            SoRenderParams & params)
{
  if (!this->backendInitialized || !this->renderTarget) {
    SoDebugError::postWarning("SoVulkanRenderManager::prepareRenderParams",
                              "backend %s, render target %s",
                              this->backendInitialized ? "initialized" : "NOT initialized",
                              this->renderTarget ? "set" : "NOT set");
    return FALSE;
  }

  SoIRRenderAction & action = this->irAction;
  action.setViewportRegion(this->viewportRegion);

  params.viewport = this->viewportRegion;
  params.viewMatrix.makeIdentity();
  params.projMatrix.makeIdentity();
  params.clearColor = this->backgroundColor;
  params.clearDepth = 1.0f;
  params.flags = 0;
  if (clearwindow || this->clearWindow) {
    params.flags |= SO_PARAM_CLEAR_WINDOW;
  }
  if (clearzbuffer || this->clearDepth) {
    params.flags |= SO_PARAM_CLEAR_DEPTH;
  }
  params.renderTarget = this->renderTarget;

  if (this->scene) {
    action.apply(this->scene);
  }
  else {
    action.beginFrame();
  }

  SoDrawList & list = action.getMutableDrawList();

  // The traversed scene state is popped when apply() returns, so harvest the
  // view/projection matrices from the first recorded command (single-camera
  // scenes are the normal retained case) instead of the action's state.
  if (list.getNumCommands() > 0) {
    const SoRenderCommand & first = list.getCommand(0);
    params.viewMatrix = first.viewMatrix;
    params.projMatrix = first.projMatrix;
  }

  list.buildSortedOrder(params.viewMatrix);
  drawlist = &list;
  return TRUE;
}

SoVulkanRenderBackend *
SoVulkanRenderManager::getBackend(void) const
{
  return &this->pimpl->backend;
}
