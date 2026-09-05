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

/*!
  \class SoShape SoShape.h Inventor/nodes/SoShape.h
  \brief The SoShape class is the superclass for geometry shapes.

  \ingroup coin_nodes

  The node types which have actual geometry to render inherit this
  class. For convenience, the SoShape class contains various common
  code used by the subclasses.
*/

// *************************************************************************

#include <Inventor/nodes/SoShape.h>

class SoVBO;
#include <Inventor/elements/SoLazyElement.h>
#include <Inventor/elements/SoMultiTextureImageElement.h>
#include <Inventor/elements/SoVertexAttributeElement.h>

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <memory>

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif // HAVE_CONFIG_H

#include <Inventor/C/glue/gl.h>
#include <Inventor/C/tidbits.h>
#include <Inventor/SbBox2f.h>
#include <Inventor/SbClip.h>
#include <Inventor/SbPlane.h>
#include <Inventor/SbTime.h>
#include <Inventor/SoPickedPoint.h>
#include <Inventor/SoPrimitiveVertex.h>
#include <Inventor/actions/SoCallbackAction.h>
#if COIN_BUILD_LEGACY_GL_RENDERER
#include <Inventor/actions/SoGLRenderAction.h>
#endif
#include <Inventor/actions/SoIRRenderAction.h>
#include <Inventor/actions/SoGetBoundingBoxAction.h>
#include <Inventor/actions/SoGetPrimitiveCountAction.h>
#include <Inventor/actions/SoRayPickAction.h>
#if COIN_BUILD_LEGACY_GL_RENDERER
#include <Inventor/annex/FXViz/elements/SoShadowStyleElement.h>
#endif
#include <Inventor/bundles/SoMaterialBundle.h>
#include <Inventor/caches/SoBoundingBoxCache.h>
#if COIN_BUILD_LEGACY_GL_RENDERER
#include <Inventor/caches/SoPrimitiveVertexCache.h>
#endif
#include <Inventor/details/SoFaceDetail.h>
#include <Inventor/details/SoLineDetail.h>
#include <Inventor/elements/SoBumpMapElement.h>
#include <Inventor/elements/SoCacheElement.h>
#include <Inventor/elements/SoComplexityElement.h>
#include <Inventor/elements/SoCoordinateElement.h>
#include <Inventor/elements/SoCullElement.h>
#include <Inventor/elements/SoGLCacheContextElement.h>
#if COIN_BUILD_LEGACY_GL_RENDERER
#include <Inventor/elements/SoGLLazyElement.h>
#endif
#if COIN_BUILD_LEGACY_GL_RENDERER
#include <Inventor/elements/SoGLMultiTextureEnabledElement.h>
#endif
#if COIN_BUILD_LEGACY_GL_RENDERER
#include <Inventor/elements/SoGLMultiTextureImageElement.h>
#endif
#if COIN_BUILD_LEGACY_GL_RENDERER
#include <Inventor/elements/SoGLShapeHintsElement.h>
#endif
#if COIN_BUILD_LEGACY_GL_RENDERER
#include <Inventor/elements/SoGLVBOElement.h>
#endif
#if COIN_BUILD_LEGACY_GL_RENDERER
#include <Inventor/elements/SoGLVertexAttributeElement.h>
#endif
#include <Inventor/elements/SoLightElement.h>
#include <Inventor/elements/SoLightModelElement.h>
#include <Inventor/elements/SoMaterialBindingElement.h>
#include <Inventor/elements/SoModelMatrixElement.h>
#include <Inventor/elements/SoMultiTextureCoordinateElement.h>
#include <Inventor/elements/SoMultiTextureEnabledElement.h>
#include <Inventor/elements/SoNormalElement.h>
#include <Inventor/elements/SoProjectionMatrixElement.h>
#include <Inventor/elements/SoShapeHintsElement.h>
#include <Inventor/elements/SoShapeStyleElement.h>
#include <Inventor/elements/SoTextureQualityElement.h>
#include <Inventor/elements/SoViewVolumeElement.h>
#include <Inventor/elements/SoViewingMatrixElement.h>
#include <Inventor/elements/SoViewportRegionElement.h>
#include <Inventor/errors/SoDebugError.h>
#if COIN_BUILD_LEGACY_GL_RENDERER
#include <Inventor/misc/SoGLBigImage.h>
#endif
#include <Inventor/misc/SoGLDriverDatabase.h>
#include <Inventor/misc/SoState.h>
#include <Inventor/nodes/SoLight.h>
#include <Inventor/nodes/SoVertexProperty.h>
#include <Inventor/nodes/SoVertexShape.h>
#include <Inventor/system/gl.h>
#include <Inventor/threads/SbMutex.h>
#include <Inventor/threads/SbStorage.h>

#ifdef HAVE_VRML97
#include <Inventor/VRMLnodes/SoVRMLIndexedFaceSet.h>
#include <Inventor/VRMLnodes/SoVRMLExtrusion.h>
#include <Inventor/VRMLnodes/SoVRMLElevationGrid.h>
#endif // HAVE_VRML97

#include "nodes/SoSubNodeP.h"
#include "rendering/SoGL.h"
#include "rendering/SoRenderIRP.h"
#include "glue/glp.h"
#include "threads/threadsutilp.h"
#include "tidbitsp.h"
#if COIN_BUILD_LEGACY_GL_RENDERER
#include "rendering/SoVBO.h"
#endif
#include "coindefs.h" // COIN_OBSOLETED()

namespace {

struct SoIRVertex {
  SbVec3f position;
  SbVec3f normal;
  SbVec4f texcoord;
  int materialIndex = 0;
};

struct SoIRBatch {
  size_t first = 0;
  size_t count = 0;
  int materialIndex = -1;

  SoIRBatch(size_t first, size_t count, int materialIndex)
    : first(first), count(count), materialIndex(materialIndex) {}
};

// Retained, shape-owned tessellation output for the IR render path.  The
// flattened stream arrays are allocated afresh on every rebuild so their
// pointers are stable across frames for an unchanged shape and change exactly
// when the shape re-tessellates.  The Vulkan backend's GPU geometry cache keys
// on these pointers, so a rebuilt cache (new pointers) forces a re-upload while
// an unchanged one (stable pointers, stable content) is safely reused.
struct SoIRRetainedGeometry {
  SoPrimitiveTopology topology = SO_TOPOLOGY_COUNT;
  std::shared_ptr<std::vector<float>> positions;
  std::shared_ptr<std::vector<float>> normals;
  std::shared_ptr<std::vector<float>> texcoords;
  // Per-vertex material index, so colors/batches can be re-resolved from the
  // current material binding/state on every emission.
  std::shared_ptr<std::vector<int>> matIndices;
  size_t vertexCount = 0;
  uint32_t vertexStride = 0;
  uint32_t texcoordStride = 0;
  uint32_t normalCount = 0;

  void invalidate()
  {
    this->topology = SO_TOPOLOGY_COUNT;
    this->positions.reset();
    this->normals.reset();
    this->texcoords.reset();
    this->matIndices.reset();
    this->vertexCount = 0;
    this->vertexStride = 0;
    this->texcoordStride = 0;
    this->normalCount = 0;
  }
};

class SoIRPrimitiveAssembler : public SoIRRenderAction::PrimitiveCollector {
public:
  SoIRPrimitiveAssembler(SoIRRenderAction * action, SoShape * shape,
                         std::vector<SoIRRetainedGeometry> * runs)
    : action(action), shape(shape), runs(runs), topology(SO_TOPOLOGY_COUNT) {}

  void onTriangle(const SoPrimitiveVertex * v1,
                  const SoPrimitiveVertex * v2,
                  const SoPrimitiveVertex * v3) override
  {
    if (this->ensureTopology(SO_TOPOLOGY_TRIANGLES)) {
      this->append(v1, v1->getMaterialIndex());
      this->append(v2, v2->getMaterialIndex());
      this->append(v3, v3->getMaterialIndex());
    }
  }

  void onLine(const SoPrimitiveVertex * v1,
              const SoPrimitiveVertex * v2) override
  {
    if (this->ensureTopology(SO_TOPOLOGY_LINES)) {
      this->append(v1, v1->getMaterialIndex());
      this->append(v2, v2->getMaterialIndex());
    }
  }

  void onPoint(const SoPrimitiveVertex * v) override
  {
    if (this->ensureTopology(SO_TOPOLOGY_POINTS)) {
      this->append(v, v->getMaterialIndex());
    }
  }

  void finalize()
  {
    this->flushRun();
  }

private:
  void flushRun()
  {
    if (this->vertices.empty()) return;

    // Flatten the accumulated SoIRVertex stream into shape-retained buffers.
    // A fresh run is created per flushRun() (one per topology run), so its
    // pointers are stable across frames for an unchanged shape and change
    // whenever the shape re-tessellates -- the property the IR/GPU caches key
    // on.  Actual command/state materialization happens later in
    // soshape_emit_ir_commands(), which runs on a cache-replay frame too.
    SoIRRetainedGeometry run;
    const size_t count = this->vertices.size();
    run.topology = this->topology;
    run.vertexCount = count;
    run.normalCount = static_cast<uint32_t>(count);
    run.vertexStride = sizeof(float) * 3;
    run.texcoordStride = sizeof(float) * 4;

    run.positions = std::make_shared<std::vector<float>>(count * 3);
    run.normals = std::make_shared<std::vector<float>>(count * 3);
    run.texcoords = std::make_shared<std::vector<float>>(count * 4);
    run.matIndices = std::make_shared<std::vector<int>>(count);

    std::vector<float> & positions = *run.positions;
    std::vector<float> & normals = *run.normals;
    std::vector<float> & texcoords = *run.texcoords;
    std::vector<int> & matIndices = *run.matIndices;
    for (size_t i = 0; i < count; ++i) {
      const SoIRVertex & vertex = this->vertices[i];
      positions[i * 3 + 0] = vertex.position[0];
      positions[i * 3 + 1] = vertex.position[1];
      positions[i * 3 + 2] = vertex.position[2];
      normals[i * 3 + 0] = vertex.normal[0];
      normals[i * 3 + 1] = vertex.normal[1];
      normals[i * 3 + 2] = vertex.normal[2];
      texcoords[i * 4 + 0] = vertex.texcoord[0];
      texcoords[i * 4 + 1] = vertex.texcoord[1];
      texcoords[i * 4 + 2] = vertex.texcoord[2];
      texcoords[i * 4 + 3] = vertex.texcoord[3];
      matIndices[i] = vertex.materialIndex;
    }

    this->runs->push_back(std::move(run));
    this->vertices.clear();
  }

  bool ensureTopology(SoPrimitiveTopology candidate)
  {
    if (this->topology == SO_TOPOLOGY_COUNT) {
      this->topology = candidate;
      return true;
    }
    if (this->topology == candidate) return true;

    this->flushRun();
    this->topology = candidate;
    return true;
  }

  void append(const SoPrimitiveVertex * vertex, int materialIndex)
  {
    SoIRVertex copy;
    copy.position = vertex->getPoint();
    copy.normal = vertex->getNormal();
    copy.texcoord = vertex->getTextureCoords();
    copy.materialIndex = materialIndex;
    this->vertices.push_back(copy);
  }

  SoIRRenderAction * action;
  SoShape * shape;
  std::vector<SoIRRetainedGeometry> * runs;
  SoPrimitiveTopology topology;
  std::vector<SoIRVertex> vertices;
};

// Build and append the SoRenderCommands for one shape's tessellated geometry.
// The positions/normals/texcoords are the shape-retained streams (stable
// pointers); the per-vertex material indices, when present, allow the
// batched/material assignment and any per-vertex colors to be re-derived from
// the *current* traversal state so a material change is always honoured.
//
// This is the materialization half of the IR render cache: it runs both on a
// fresh tessellation (right after generatePrimitives) and on a cache-replay
// frame (without re-running generatePrimitives), and is the only per-frame
// work remaining for an unchanged shape.
static void
soshape_emit_ir_commands(SoIRRenderAction * action, SoShape * shape,
                         SoState * state, const SoIRRetainedGeometry & geom,
                         bool retained, std::vector<SoIRBatch> & batchScratch)
{
  const std::vector<float> & positions = *geom.positions;
  const std::vector<float> & normals = *geom.normals;
  const std::vector<float> & texcoords = *geom.texcoords;
  const bool hasMatIndices = geom.matIndices != nullptr;
  static const std::vector<int> emptyMatIndices;
  const std::vector<int> & matIndices =
    hasMatIndices ? *geom.matIndices : emptyMatIndices;
  const size_t count = geom.vertexCount;
  const size_t primitiveWidth = geom.topology == SO_TOPOLOGY_TRIANGLES ? 3
    : geom.topology == SO_TOPOLOGY_LINES ? 2 : 1;

  const SoMaterialBindingElement::Binding materialBinding =
    SoMaterialBindingElement::get(state);
  const bool hasExplicitMaterialIndices =
    materialBinding == SoMaterialBindingElement::PER_PART ||
    materialBinding == SoMaterialBindingElement::PER_PART_INDEXED ||
    materialBinding == SoMaterialBindingElement::PER_FACE ||
    materialBinding == SoMaterialBindingElement::PER_FACE_INDEXED ||
    materialBinding == SoMaterialBindingElement::PER_VERTEX_INDEXED;
  const bool hasPerVertexMaterials =
    materialBinding == SoMaterialBindingElement::PER_VERTEX ||
    materialBinding == SoMaterialBindingElement::PER_VERTEX_INDEXED;

  // Reuse the shape-owned scratch (capacity persists across frames) rather
  // than allocating a fresh batch vector per run.
  std::vector<SoIRBatch> & batches = batchScratch;
  batches.clear();
  batches.reserve((count + primitiveWidth - 1) / primitiveWidth);
  bool hasMixedMaterials = false;
  if (!hasExplicitMaterialIndices) {
    batches.push_back(SoIRBatch(0, count, 0));
  }
  for (size_t first = 0; hasExplicitMaterialIndices && first < count;) {
    const size_t primitiveCount = std::min(primitiveWidth, count - first);
    int materialIndex = hasMatIndices ? matIndices[first] : 0;
    for (size_t i = 1; i < primitiveCount; ++i) {
      if ((hasMatIndices ? matIndices[first + i] : 0) != materialIndex) {
        materialIndex = -1;
        hasMixedMaterials = true;
        break;
      }
    }
    if (batches.empty() || batches.back().materialIndex != materialIndex) {
      batches.push_back(SoIRBatch(first, primitiveCount, materialIndex));
    }
    else {
      batches.back().count += primitiveCount;
    }
    first += primitiveCount;
  }

  // Per-vertex/mixed colors depend on the current material state, so they are
  // resolved into action-owned frame storage on every emission.  This is only
  // reached for the (rare) explicit/per-vertex binding case; the common
  // single-material case keeps colors null and is shaded by the material
  // uniform.  The colors buffer lives in the action's arena (persists through
  // the backend's per-frame update) rather than a local vector, and any command
  // carrying colors is deliberately not claimed `retained`, since the arena
  // rewrites the same pointer in place.
  float * colors = nullptr;
  if (hasMixedMaterials || hasPerVertexMaterials) {
    colors = static_cast<float *>(
      action->allocateGeometryStorage(sizeof(float) * 4 * count));
    for (size_t i = 0; i < count; ++i) {
      const int materialIndex = std::max(hasMatIndices ? matIndices[i] : 0, 0);
      const SbColor & color =
        SoLazyElement::getDiffuse(state, materialIndex);
      const float alpha = 1.0f - SoLazyElement::getTransparency(state, materialIndex);
      colors[i * 4 + 0] = color[0];
      colors[i * 4 + 1] = color[1];
      colors[i * 4 + 2] = color[2];
      colors[i * 4 + 3] = alpha;
    }
  }

  // Validate the clip-debug flag once: it is process-lifetime and this runs on
  // the per-command path, so a getenv() (environ scan) per command is pure
  // overhead.  Mirrors SoVulkanRenderManager's clipDebugEnabled().
  static const bool clipDebug = std::getenv("FC_VULKAN_CLIP_DEBUG") != nullptr;

  for (const SoIRBatch & batch : batches) {
    SoRenderCommand command = {};
    command.geometry.topology = geom.topology;
    command.geometry.vertexCount = static_cast<uint32_t>(batch.count);
    command.geometry.normalCount = command.geometry.vertexCount;
    command.geometry.vertexStride = sizeof(float) * 3;
    command.geometry.texcoordStride = sizeof(float) * 4;
    command.geometry.positions = positions.data() + batch.first * 3;
    command.geometry.normals = normals.data() + batch.first * 3;
    command.geometry.texcoords = texcoords.data() + batch.first * 4;
    command.geometry.colors = colors ? colors + batch.first * 4 : nullptr;
    // Co-own the shape-retained stream buffers: SoShape::notify() may drop
    // the shape's own reference (and free the chunk) on any field change, on
    // another thread, before this command is re-emitted with a fresh buffer.
    // The command's ownership keeps the storage -- and thus the raw pointers
    // above -- valid until the command itself is replaced.
    command.geometry.positionOwner = geom.positions;
    command.geometry.normalOwner = geom.normals;
    command.geometry.texcoordOwner = geom.texcoords;
    // The position/normal/texcoord streams are the shape-retained buffers
    // (stable pointers, new pointer on change), so the backend can trust
    // pointer identity without re-hashing.  Commands carrying per-vertex colors
    // are NOT claimed retained: their colors come from the per-frame arena and
    // could be rewritten in place, so those must still be content-verified.
    command.geometry.retained = retained && (colors == nullptr);
    command.modelMatrix = SoModelMatrixElement::get(state);
    command.viewMatrix = SoViewingMatrixElement::get(state);
    command.projMatrix = SoProjectionMatrixElement::get(state);
    if (clipDebug) {
      static int mmLog = 0;
      if (mmLog++ < 6) {
        SbBool isId = FALSE;
        const SbMatrix & el = SoModelMatrixElement::get(state, isId);
        SbMatrix mm = command.modelMatrix;
        fprintf(stderr, "[SHAPE] cmd rec pass=%d verts=%u model00=%.3f m11=%.3f "
                        "m22=%.3f trans=(%.3f,%.3f,%.3f) isIdentity=%d "
                        "el00=%.3f eltrans=(%.3f,%.3f,%.3f)\n",
                static_cast<int>(command.pass),
                static_cast<unsigned>(command.geometry.vertexCount),
                mm[0][0], mm[1][1], mm[2][2],
                mm[3][0], mm[3][1], mm[3][2],
                isId ? 1 : 0, el[0][0], el[3][0], el[3][1], el[3][2]);
      }
    }
    SoRenderIR::fillMaterialFromState(
      state, command.material, std::max(batch.materialIndex, 0));
    command.material.vertexColorAlphaIncludesOpacity =
      command.geometry.colors != nullptr;
    SoRenderIR::fillTextureFromState(state, action, command.material);
    SoRenderIR::fillRenderStateFromState(state, command.state);
    SoRenderIR::ensureMaterialBlendState(command.state, command.material);
    bool transparent = SoRenderIR::isMaterialTransparent(command.material);
    if (!transparent && command.geometry.colors) {
      for (uint32_t i = 0; i < command.geometry.vertexCount; ++i) {
        if (command.geometry.colors[i * 4 + 3] < 0.999f) {
          transparent = true;
          break;
        }
      }
    }
    if (!transparent && command.state.blend.enabled
        && !command.state.depth.writeEnabled
        && command.state.depth.func == SO_DEPTH_ALWAYS) {
      transparent = true;
    }
    command.pass = transparent ? SO_RENDERPASS_TRANSPARENT
                               : SO_RENDERPASS_OPAQUE;
    command.lightingHandle = SoRenderIR::fillLightingFromState(
      state, action->getMutableDrawList());
    command.userData = shape;
    action->getMutableDrawList().addCommand(command);
  }
}

}

// SoShape.cpp grew too big, so I had to move some code into new
// files. pederb, 2001-07-18
#include "soshape_primdata.h"
#if COIN_BUILD_LEGACY_GL_RENDERER
#include "soshape_trianglesort.h"
#include "soshape_bigtexture.h"
#include "soshape_bumprender.h"
#endif

// *************************************************************************

/*!
  \enum SoShape::TriangleShape
  \COININTERNAL
*/

/*!
  \fn void SoShape::computeBBox(SoAction * action, SbBox3f & box, SbVec3f & center)

  Implemented by SoShape subclasses to let the SoShape superclass know
  the exact size and weighted center point of the shape's bounding box.

  The bounding box and center point should be calculated and returned
  in the local coordinate system.

  The method implements action behavior for shape nodes for
  SoGetBoundingBoxAction. It is invoked from
  SoShape::getBoundingBox(). (Subclasses should \e not override
  SoNode::getBoundingBox().)

  The \a box parameter sent in is guaranteed to be an empty box, while
  \a center is undefined upon function entry.
*/

/*!
  \fn void SoShape::generatePrimitives(SoAction * action)

  The method implements action behavior for shape nodes for
  SoCallbackAction. It is invoked from
  SoShape::callback(). (Subclasses should \e not override
  SoNode::callback().)

  The subclass implementations uses the convenience methods
  SoShape::beginShape(), SoShape::shapeVertex(), and
  SoShape::endShape(), with SoDetail instances, to pass the primitives
  making up the shape back to the caller.
*/

// *************************************************************************

class SoShapeP {
public:
  SoShapeP() {
    this->bboxcache = NULL;
#if COIN_BUILD_LEGACY_GL_RENDERER
    this->pvcache = NULL;
#endif
#if COIN_BUILD_LEGACY_GL_RENDERER
    this->bumprender = NULL;
#endif
    this->rendercnt = 0;
    this->flags = 0;
    this->irCacheValid = false;
  }
  ~SoShapeP() {
    if (this->bboxcache) { this->bboxcache->unref(); }
#if COIN_BUILD_LEGACY_GL_RENDERER
    if (this->pvcache) { this->pvcache->unref(); }
#endif
#if COIN_BUILD_LEGACY_GL_RENDERER
    delete this->bumprender;
#endif
  }
  enum {
    RENDERCNT_BITS = 4,     // bits needed to store rendercnt
    FLAG_BITS = 4           // bits needed to store flags
  };
  enum Flags {
    SHOULD_BBOX_CACHE = 0x1,
    NEED_SETUP_SHAPE_HINTS = 0x2,
    DISABLE_VERTEX_ARRAY_CACHE = 0x4,
  };

  static void calibrateBBoxCache(void);
  static double bboxcachetimelimit;
  SoBoundingBoxCache * bboxcache;
#if COIN_BUILD_LEGACY_GL_RENDERER
  SoPrimitiveVertexCache * pvcache;
#endif
#if COIN_BUILD_LEGACY_GL_RENDERER
  soshape_bumprender * bumprender;
#endif
  uint32_t flags : FLAG_BITS;
  // stores the number of frames rendered with no node changes
  uint32_t rendercnt : RENDERCNT_BITS;

  // Retained IR tessellation output (positions/normals/texcoords + per-vertex
  // material indices), cached so an unchanged shape skips generatePrimitives()
  // on the IR/Vulkan path.  `irCacheValid` is cleared in SoShape::notify() on
  // any field change; the build-time complexity is snapshotted so a change in
  // the (non-notified) Complexity element -- which drives primitive count for
  // shapes like SoSphere -- also forces a rebuild rather than serving stale
  // tessellation.  A fresh build reallocates the stream buffers (new pointers)
  // so the backend geometry cache detects the change.
  std::vector<SoIRRetainedGeometry> irRuns;
  bool irCacheValid;
  float irCacheComplexity = -1.0f;

  // Reusable scratch for the IR command emitter: the batch vector is rebuilt
  // (and only transiently used) on every run/frame, so keeping its capacity
  // across frames avoids a small heap allocation per emission.
  std::vector<SoIRBatch> irBatchScratch;

  // needed since some VRML97 nodes change the GL state inside the node
  void testSetupShapeHints(SoShape * shape) {
#ifdef HAVE_VRML97
    if ((this->flags & SoShapeP::NEED_SETUP_SHAPE_HINTS) == 0) {
      if (shape->isOfType(SoVRMLIndexedFaceSet::getClassTypeId()) ||
          shape->isOfType(SoVRMLExtrusion::getClassTypeId()) ||
          shape->isOfType(SoVRMLElevationGrid::getClassTypeId())) {
        this->flags |= SoShapeP::NEED_SETUP_SHAPE_HINTS;
      }
    }
#endif // HAVE_VRML97
  }
  void setupShapeHints(SoShape * shape, SoState * state) {
#ifdef HAVE_VRML97
    if (this->flags & SoShapeP::NEED_SETUP_SHAPE_HINTS) {
      SbBool ccw = ((SoSFBool*)(shape->getField("ccw")))->getValue();
      SbBool solid = ((SoSFBool*)(shape->getField("solid")))->getValue();
#if COIN_BUILD_LEGACY_GL_RENDERER
      SoGLShapeHintsElement::forceSend(state, ccw, solid, !solid);
#else
      (void)state;
#endif
    }
#endif // HAVE_VRML97
  }

  // we can use a per-instance mutex here instead of this class-wide
  // one, but we go for the class-wide one since at least Microsoft Windows
  // might have a rather strict limit on the total amount of mutex
  // resources a process / user can hold at any one time.
  //
  // i haven't looked too hard at the locked code regions, however --
  // it might be that a class-wide lock can cause significantly less
  // efficient execution in a multi-threaded environment. if so, we
  // will have to come up with something smarter than this (a mutex
  // pool or something, i suppose).
  //
  // -mortene.
  static SbMutex * mutex;

#ifdef COIN_THREADSAFE
  void lock(void) { SoShapeP::mutex->lock(); }
  void unlock(void) { SoShapeP::mutex->unlock(); }
#else // ! COIN_THREADSAFE
  void lock(void) { }
  void unlock(void) { }
#endif // ! COIN_THREADSAFE

  static void cleanup(void);
};

double SoShapeP::bboxcachetimelimit;

SbMutex * SoShapeP::mutex = NULL;

#undef PRIVATE
#define PRIVATE(p) ((p)->pimpl)

// *************************************************************************
// code/structures to handle static and/or thread safe data

enum SoShapeRenderMode {
  NORMAL,
  BIGTEXTURE,
  SORTED_TRIANGLES,
  PVCACHE
};

typedef struct {
  soshape_primdata * primdata;
#if COIN_BUILD_LEGACY_GL_RENDERER
  SbList <soshape_bigtexture*> * bigtexturelist;
  SbList <uint32_t> * bigtexturecontext;
  soshape_trianglesort * trianglesort;

  soshape_bigtexture * currentbigtexture;
#endif
  // used in generatePrimitives() callbacks to set correct material
  SoMaterialBundle * currentbundle;

#if COIN_BUILD_LEGACY_GL_RENDERER
  int rendermode;
#endif
} soshape_staticdata;

#if COIN_BUILD_LEGACY_GL_RENDERER
static soshape_bigtexture *
soshape_get_bigtexture(soshape_staticdata * data, uint32_t context)
{
  for (int i = 0; i < data->bigtexturecontext->getLength(); i++) {
    if ((*(data->bigtexturecontext))[i] == context) {
      return (*(data->bigtexturelist))[i];
    }
  }
  soshape_bigtexture * newtex = new soshape_bigtexture;
  data->bigtexturelist->append(newtex);
  data->bigtexturecontext->append(context);
  return newtex;
}
#endif

static void
soshape_construct_staticdata(void * closure)
{
  soshape_staticdata * data = (soshape_staticdata*) closure;

#if COIN_BUILD_LEGACY_GL_RENDERER
  data->bigtexturelist = new SbList <soshape_bigtexture*>;
  data->bigtexturecontext = new SbList <uint32_t>;
  data->trianglesort = new soshape_trianglesort();
  data->rendermode = NORMAL;
#endif
  data->primdata = new soshape_primdata();
}

static void
soshape_destruct_staticdata(void * closure)
{
  soshape_staticdata * data = (soshape_staticdata*) closure;
#if COIN_BUILD_LEGACY_GL_RENDERER
  for (int i = 0; i < data->bigtexturelist->getLength(); i++) {
    delete (*(data->bigtexturelist))[i];
  }
  delete data->bigtexturelist;
  delete data->bigtexturecontext;
#endif
  delete data->primdata;
#if COIN_BUILD_LEGACY_GL_RENDERER
  delete data->trianglesort;
#endif
}

static SbStorage * soshape_staticstorage;

static soshape_staticdata *
soshape_get_staticdata(void)
{
  return (soshape_staticdata*) soshape_staticstorage->get();
}

// called by atexit
void
SoShapeP::cleanup(void)
{
  delete soshape_staticstorage;
  soshape_staticstorage = NULL;

  delete SoShapeP::mutex;
  SoShapeP::mutex = NULL;
}

// *************************************************************************

SO_NODE_ABSTRACT_SOURCE(SoShape);

// *************************************************************************

/*!
  Constructor.
*/
SoShape::SoShape(void)
{
  SO_NODE_INTERNAL_CONSTRUCTOR(SoShape);
  PRIVATE(this) = new SoShapeP;
}

/*!
  Destructor.
*/
SoShape::~SoShape()
{
  delete PRIVATE(this);
}

#if COIN_BUILD_LEGACY_GL_RENDERER
SbBool
SoShape::canRenderSortedTriangles(void) const
{
  return TRUE;
}
#endif

/*!
  \copybrief SoBase::initClass(void)
*/
void
SoShape::initClass(void)
{
  SO_NODE_INTERNAL_INIT_ABSTRACT_CLASS(SoShape, SO_FROM_INVENTOR_1);

#ifdef COIN_THREADSAFE
  SoShapeP::mutex = new SbMutex;
#endif // COIN_THREADSAFE

  soshape_staticstorage =
    new SbStorage(sizeof(soshape_staticdata),
                  soshape_construct_staticdata,
                  soshape_destruct_staticdata);
  SoShapeP::calibrateBBoxCache();

  coin_atexit((coin_atexit_f *)SoShapeP::cleanup, CC_ATEXIT_NORMAL);
}

// Doc in parent.
void
SoShape::getBoundingBox(SoGetBoundingBoxAction * action)
{
  SbBox3f box;
  SbVec3f center;
  this->getBBox(action, box, center);
  if (!box.isEmpty()) {
    action->extendBy(box);
    action->setCenter(center, TRUE);
  }
}

// Doc in parent.
#if COIN_BUILD_LEGACY_GL_RENDERER
void
SoShape::GLRender(SoGLRenderAction * action)
{
  // if we get here, the shape do not have a render method and
  // generatePrimitives should therefore be used to render the
  // shape. This is probably painfully slow, so if you want speed,
  // implement the GLRender() method.  pederb, 20000612

  if (!this->shouldGLRender(action)) return;

  // test for SoVertexShape node and push data onto the state before
  // calling generatePrimitives(). This is needed for SoMaterialBundle
  // to work correctly.
  SoVertexProperty * vp = NULL;
  if (this->isOfType(SoVertexShape::getClassTypeId())) {
    vp = (SoVertexProperty*) ((SoVertexShape*)this)->vertexProperty.getValue();
  }

  if (vp) {
    action->getState()->push();
    vp->doAction(action);
  }
  SoMaterialBundle mb(action);
  mb.sendFirst();
  soshape_get_staticdata()->currentbundle = &mb;  // needed in the primitive callbacks
  this->generatePrimitives(action);

  if (vp) action->getState()->pop();
}
#endif

void
SoShape::IRRender(SoIRRenderAction * action)
{
  if (!action) return;
  if (getenv("FC_IR_BREADCRUMB")) {
    static int n = 0;
    if (n++ < 10) fprintf(stderr, "[BC-IR] IRRender shape=%p type=%s\n",
                          (void *)this,
                          this->getTypeId().getName().getString());
  }

  SoState * state = action->getState();

  const SoShapeStyleElement * shapestyle = SoShapeStyleElement::get(state);
  const unsigned int shapestyleflags = shapestyle->getFlags();
  if (shapestyleflags & SoShapeStyleElement::INVISIBLE) return;

  // Draw style BOUNDS: record the shape's local bounding box as a solid
  // cube instead of the shape's primitives, mirroring GLRenderBoundingBox().
  if (shapestyleflags & SoShapeStyleElement::BBOXCMPLX) {
    SbBox3f box;
    SbVec3f center;
    this->getBBox(action, box, center);
    if (box.isEmpty()) return;

    const SbVec3f min = box.getMin();
    const SbVec3f max = box.getMax();
    const SbVec3f c[8] = {
      SbVec3f(min[0], min[1], min[2]),
      SbVec3f(max[0], min[1], min[2]),
      SbVec3f(max[0], max[1], min[2]),
      SbVec3f(min[0], max[1], min[2]),
      SbVec3f(min[0], min[1], max[2]),
      SbVec3f(max[0], min[1], max[2]),
      SbVec3f(max[0], max[1], max[2]),
      SbVec3f(min[0], max[1], max[2])
    };
    static const int faces[6][4] = {
      {4, 5, 6, 7},  // +Z
      {0, 3, 2, 1},  // -Z
      {5, 1, 2, 6},  // +X
      {0, 4, 7, 3},  // -X
      {7, 6, 2, 3},  // +Y
      {0, 1, 5, 4}   // -Y
    };
    static const SbVec3f normals[6] = {
      SbVec3f(0.0f, 0.0f, 1.0f),
      SbVec3f(0.0f, 0.0f, -1.0f),
      SbVec3f(1.0f, 0.0f, 0.0f),
      SbVec3f(-1.0f, 0.0f, 0.0f),
      SbVec3f(0.0f, 1.0f, 0.0f),
      SbVec3f(0.0f, -1.0f, 0.0f)
    };

    // The bounds cube is tiny and its geometry follows the (per-frame) bbox,
    // so it is generated each time (never cached) and emitted immediately.
    SoIRRetainedGeometry boxRun;
    {
      std::vector<SoIRRetainedGeometry> boxRuns;
      SoIRPrimitiveAssembler assembler(action, this, &boxRuns);
      action->pushPrimitiveCollector(&assembler);
      for (int f = 0; f < 6; ++f) {
        SoPrimitiveVertex v[4];
        for (int i = 0; i < 4; ++i) {
          v[i].setPoint(c[faces[f][i]]);
          v[i].setNormal(normals[f]);
        }
        assembler.onTriangle(&v[0], &v[1], &v[2]);
        assembler.onTriangle(&v[0], &v[2], &v[3]);
      }
      action->popPrimitiveCollector(&assembler);
      assembler.finalize();
      if (!boxRuns.empty()) {
        boxRun = std::move(boxRuns[0]);
      }
    }
    if (boxRun.positions) {
      soshape_emit_ir_commands(action, this, state, boxRun, false,
                               PRIVATE(this)->irBatchScratch);
    }
    return;
  }

  SoVertexProperty * vertexProperty =
    this->isOfType(SoVertexShape::getClassTypeId())
      ? (SoVertexProperty *) static_cast<SoVertexShape *>(this)->vertexProperty.getValue()
      : NULL;
  if (vertexProperty) {
    state->push();
    vertexProperty->doAction(action);
  }

  // IR render cache: retain the tessellation output so an unchanged shape does
  // not re-run generatePrimitives() on every frame.  Material/state-dependent
  // work (batches, per-vertex colors, matrices, lighting) is re-derived on
  // every emission from the current state, so only the (expensive) geometry
  // generation is skipped.  The cache is invalidated on any field change via
  // SoShape::notify(); a rebuild allocates fresh stream buffers (new pointers)
  // so the backend geometry cache observes the change.
  //
  // notify() and IRRender may run on different threads (GUI vs render), so the
  // shared irRuns/irCacheValid are guarded by the shape mutex.  geometry
  // generation on a missed cache runs unlocked into a local buffer; only the
  // cheap swap/snapshot is under the lock.  Emitting from a snapshot of the
  // retained runs keeps the shared_ptr stream buffers alive even if a
  // concurrent edit invalidates the cache mid-frame.
  std::vector<SoIRRetainedGeometry> emitRuns;
  const float complexity = SoComplexityElement::get(state);
  PRIVATE(this)->lock();
  const bool cacheValid =
    PRIVATE(this)->irCacheValid && !PRIVATE(this)->irRuns.empty() &&
    PRIVATE(this)->irCacheComplexity == complexity;
  if (cacheValid) {
    emitRuns = PRIVATE(this)->irRuns;
  }
  PRIVATE(this)->unlock();

  if (!cacheValid) {
    std::vector<SoIRRetainedGeometry> built;
    SoIRPrimitiveAssembler assembler(action, this, &built);
    action->pushPrimitiveCollector(&assembler);
    this->generatePrimitives(action);
    action->popPrimitiveCollector(&assembler);
    assembler.finalize();
    PRIVATE(this)->lock();
    PRIVATE(this)->irRuns.swap(built);
    PRIVATE(this)->irCacheValid = !PRIVATE(this)->irRuns.empty();
    PRIVATE(this)->irCacheComplexity = complexity;
    emitRuns = PRIVATE(this)->irRuns;
    PRIVATE(this)->unlock();
  }

  // Emitting writes the shape-owned batch scratch.  Guard it with the same
  // mutex that protects the cache: notify() and IRRender may run on different
  // threads, and a second IRRender (e.g. a shared shape rendered by another
  // viewport's render manager) would otherwise race on irBatchScratch.  The
  // retained geometry itself is safe to read unlocked (emitRuns is a snapshot
  // holding shared_ptr refs), so this only serializes the scratch.
  PRIVATE(this)->lock();
  for (const SoIRRetainedGeometry & run : emitRuns) {
    soshape_emit_ir_commands(action, this, state, run, true,
                             PRIVATE(this)->irBatchScratch);
  }
  PRIVATE(this)->unlock();

  if (vertexProperty) state->pop();
}

// Doc in parent.
void
SoShape::callback(SoCallbackAction * action)
{
  if (action->shouldGeneratePrimitives(this)) {
    soshape_staticdata * shapedata = soshape_get_staticdata();
    shapedata->primdata->faceCounter = 0;
    this->generatePrimitives(action);
  }
}

// test bbox intersection
static SbBool
soshape_ray_intersect(SoRayPickAction * action, const SbBox3f & box)
{
  if (box.isEmpty()) return FALSE;
  return action->intersect(box, TRUE);
}


/*!
  Calculates picked point based on primitives generated by subclasses.
*/
void
SoShape::rayPick(SoRayPickAction * action)
{
  if (this->shouldRayPick(action)) {
    this->computeObjectSpaceRay(action);

    if (!PRIVATE(this)->bboxcache ||
        !PRIVATE(this)->bboxcache->isValid(action->getState()) ||
        soshape_ray_intersect(action, PRIVATE(this)->bboxcache->getProjectedBox())) {
      this->generatePrimitives(action);
    }
  }
}

/*!
  A convenience function that returns the size of a \a boundingbox
  projected onto the screen. Useful for \c SCREEN_SPACE complexity
  geometry.
*/
void
SoShape::getScreenSize(SoState * const state, const SbBox3f & boundingbox,
                       SbVec2s & rectsize)
{
  SbMatrix projmatrix;
  projmatrix = (SoModelMatrixElement::get(state) *
                SoViewingMatrixElement::get(state) *
                SoProjectionMatrixElement::get(state));

  SbVec2s vpsize = SoViewportRegionElement::get(state).getViewportSizePixels();
  if (boundingbox.isEmpty())
  {
      rectsize = vpsize * 0.5;
      return;
  }
  SbVec3f bmin, bmax;
  boundingbox.getBounds(bmin, bmax);

  SbVec3f v;
  SbBox2f normbox;
  normbox.makeEmpty();
  for (int i = 0; i < 8; i++) {
    v.setValue(i&1 ? bmin[0] : bmax[0],
               i&2 ? bmin[1] : bmax[1],
               i&4 ? bmin[2] : bmax[2]);
    projmatrix.multVecMatrix(v, v);
    normbox.extendBy(SbVec2f(v[0], v[1]));
  }
  float nx, ny;
  normbox.getSize(nx, ny);

  // restrict size of projection. It is often way off when object
  // intersects the near plane. We should probably do clipping against
  // the view volume do be 100% correct, but that would be too slow.
  // pederb, 2001-05-20
  if (nx > 10.0f) nx = 10.0f;
  if (ny > 10.0f) ny = 10.0f;

  rectsize[0] = (short) SbMin(32767.0f, float(vpsize[0])*0.5f*nx);
  rectsize[1] = (short) SbMin(32767.0f, float(vpsize[1])*0.5f*ny);
}

/*!
  Returns the complexity value to be used by subclasses. Considers
  complexity type. For \c OBJECT_SPACE complexity this will be a
  number between 0 and 1. For \c SCREEN_SPACE complexity it is a
  number from 0 and up.
*/
float
SoShape::getComplexityValue(SoAction * action)
{
  SoState * state = action->getState();
  switch (SoComplexityTypeElement::get(state)) {
  case SoComplexityTypeElement::SCREEN_SPACE:
    {
      SbBox3f box;
      SbVec3f center;
      this->getBBox(action, box, center);
      SbVec2s size;
      SoShape::getScreenSize(state, box, size);
      // FIXME: probably needs calibration.

#if 1 // testing new complexity code
      // The cast within the sqrt() is done to avoid ambigouity error
      // from HPUX aCC, as sqrt() can be either "long double sqrt(long
      // double)" or "float sqrt(float)". mortene.
      return float(sqrt((float)SbMax(size[0], size[1]))) * 0.4f *
        SoComplexityElement::get(state);
#else // first version
      float numPixels = float(size[0])* float(size[1]);
      return numPixels * 0.0001f * SoComplexityElement::get(state);
#endif
    }
  case SoComplexityTypeElement::OBJECT_SPACE:
    return SoComplexityElement::get(state);
  case SoComplexityTypeElement::BOUNDING_BOX:
    // return default value. We might get here when generating
    // primitives, not when rendering.
    return 0.5f;
  default:
    assert(0 && "unknown complexity type");
    return 0.5f;
  }
}

#if COIN_BUILD_LEGACY_GL_RENDERER
/*!
  \COININTERNAL
*/
SbBool
SoShape::shouldGLRender(SoGLRenderAction * action)
{
  SoState * state = action->getState();

  const SoShapeStyleElement * shapestyle = SoShapeStyleElement::get(state);
  unsigned int shapestyleflags = shapestyle->getFlags();

  if (shapestyleflags & SoShapeStyleElement::INVISIBLE)
    return FALSE;

  if (PRIVATE(this)->bboxcache && !state->isCacheOpen() && !SoCullElement::completelyInside(state)) {
    if (PRIVATE(this)->bboxcache->isValid(state)) {
      if (SoCullElement::cullTest(state, PRIVATE(this)->bboxcache->getProjectedBox())) {
        return FALSE;
      }
    }
  }

  SbBool transparent = (shapestyleflags & (SoShapeStyleElement::TRANSP_TEXTURE|
                                           SoShapeStyleElement::TRANSP_MATERIAL)) != 0;

  if (shapestyleflags & SoShapeStyleElement::SHADOWMAP) {
    if (transparent) return FALSE;
    int style = SoShadowStyleElement::get(state);
    if (style & SoShadowStyleElement::CASTS_SHADOW) return TRUE;
    return FALSE;
  }

  if (action->handleTransparency(transparent))
    return FALSE;

  if (shapestyleflags & SoShapeStyleElement::BBOXCMPLX) {
    this->GLRenderBoundingBox(action);
    return FALSE;
  }

  // test if we should sort triangles before rendering
  if (transparent && this->canRenderSortedTriangles() &&
      (shapestyleflags & SoShapeStyleElement::TRANSP_SORTED_TRIANGLES)) {
    // lock since pvcache is shared among all threads
    PRIVATE(this)->lock();
    this->validatePVCache(action);

    int arrays = SoPrimitiveVertexCache::NORMAL|SoPrimitiveVertexCache::COLOR;
    SoGLMultiTextureImageElement::Model model;
    SbColor blendcolor;
    SoGLImage * glimage = SoGLMultiTextureImageElement::get(state, 0, model, blendcolor);
    if (glimage) arrays |= SoPrimitiveVertexCache::TEXCOORD;

    SoMaterialBundle mb(action);
    mb.sendFirst();
    PRIVATE(this)->setupShapeHints(this, state);
    PRIVATE(this)->pvcache->depthSortTriangles(state);
    PRIVATE(this)->pvcache->renderTriangles(state, arrays);
    if (PRIVATE(this)->pvcache->getNumLineIndices() ||
        PRIVATE(this)->pvcache->getNumPointIndices()) {
      const SoNormalElement * nelem = SoNormalElement::getInstance(state);
      if (nelem->getNum() == 0) {
        glPushAttrib(GL_LIGHTING_BIT);
        glDisable(GL_LIGHTING);
        arrays &= SoPrimitiveVertexCache::NORMAL;
      }
      PRIVATE(this)->pvcache->renderLines(state, arrays);
      PRIVATE(this)->pvcache->renderPoints(state, arrays);

      if (nelem->getNum() == 0) {
        glPopAttrib();
      }
    }
    PRIVATE(this)->unlock();
    return FALSE; // tell shape _not_ to render
  }

  if (shapestyleflags & SoShapeStyleElement::BIGIMAGE) {
    SoGLMultiTextureImageElement::Model model;
    SbColor blendcolor;
    SoGLImage * glimage = SoGLMultiTextureImageElement::get(state, 0, model, blendcolor);
    if (glimage &&
        glimage->isOfType(SoGLBigImage::getClassTypeId()) &&
        SoGLMultiTextureEnabledElement::get(state, 0)) {

      // don't attempt to cache bigimage shapes
      if (state->isCacheOpen()) {
        SoCacheElement::invalidate(state);
      }
      SoGLCacheContextElement::shouldAutoCache(state,
                                               SoGLCacheContextElement::DONT_AUTO_CACHE);

      soshape_staticdata * shapedata = soshape_get_staticdata();

      // do this before generating triangles to get correct
      // material for lines and point (only triangles are handled for now).
      SoMaterialBundle mb(action);
      mb.sendFirst();
      shapedata->currentbundle = &mb;

      SoGLBigImage * big = (SoGLBigImage*) glimage;

      shapedata->rendermode = BIGTEXTURE;

      soshape_bigtexture * bigtex = soshape_get_bigtexture(shapedata, action->getCacheContext());
      shapedata->currentbigtexture = bigtex;
      bigtex->beginShape(big, SoTextureQualityElement::get(state));
      this->generatePrimitives(action);
      // endShape() returns whether more/less detailed textures need to be
      // fetched. We force a redraw if this is needed.
      if (bigtex->endShape(state, this, mb) == FALSE) {
        action->getCurPath()->getHead()->touch();
      }
      shapedata->rendermode = NORMAL;

      return FALSE;
    }
  }

  const cc_glglue * glue = sogl_glue_instance(state);

  if (shapestyleflags & SoShapeStyleElement::BUMPMAP) {
    const SoNodeList & lights = SoLightElement::getLights(state);
    if (lights.getLength()) {
      // lock since bumprender and pvcache is shared among all threads
      PRIVATE(this)->lock();
      if (PRIVATE(this)->bumprender == NULL) {
        PRIVATE(this)->bumprender = new soshape_bumprender;
      }
      this->validatePVCache(action);
      if (PRIVATE(this)->pvcache->getNumTriangleIndices() == 0) {
        PRIVATE(this)->unlock();
        return TRUE;
      }
      SoGLLazyElement::getInstance(state)->send(state, SoLazyElement::ALL_MASK);

      glPushAttrib(GL_DEPTH_BUFFER_BIT);
      glDepthFunc(GL_LEQUAL);
      glDisable(GL_LIGHTING);

      glColor3f(1.0f, 1.0f, 1.0f);
      PRIVATE(this)->setupShapeHints(this, state);
      const int numlights = lights.getLength();
      for (int i = 0; i < numlights; i++) {
        // fetch matrix that convert the light from its object space
        // to the OpenGL world space
        SbMatrix lm = SoLightElement::getMatrix(state, i);

        // convert light back to this objects' object space
        SbMatrix m = SoModelMatrixElement::get(state) *
          SoViewingMatrixElement::get(state);
        m = m.inverse();
        m.multLeft(lm);


        // bumprender is shared among all threads, so we need to lock
        // when we get here since some internal arrays are used while
        // rendering
        //
        // FIXME: about the above comment; i don't see any locking...?
        // -mortene.
        PRIVATE(this)->bumprender->renderBump(state, PRIVATE(this)->pvcache,
                                              (SoLight*) lights[i], m);

        if (i == 0) glEnable(GL_BLEND);
        if (i == numlights-1) {
          glBlendFunc(GL_DST_COLOR, GL_ZERO);
        }
        else if (i == 0) {
          glBlendFunc(GL_ONE, GL_ONE);
        }
      }


      SoGLLazyElement::getInstance(state)->reset(state,
                                                 SoLazyElement::DIFFUSE_MASK);
      SoMaterialBundle mb(action);
      mb.sendFirst();
      PRIVATE(this)->setupShapeHints(this, state);
      PRIVATE(this)->bumprender->renderNormal(state, PRIVATE(this)->pvcache);

      const SbColor spec = SoLazyElement::getSpecular(state);
      if (spec[0] != 0 || spec[1] != 0 || spec[2] != 0) { // Is the spec. color black?

        // Can the hardware do specular bump maps?
        if (glue->has_arb_fragment_program &&
            glue->has_arb_vertex_program) {

          SoGLLazyElement::getInstance(state)->reset(state,
                                                     SoLazyElement::DIFFUSE_MASK);
          glEnable(GL_BLEND);
          glBlendFunc(GL_ONE, GL_ONE);

          for (int i = 0; i < numlights; i++) {
            SbMatrix lm = SoLightElement::getMatrix(state, i);
            SbMatrix m = SoModelMatrixElement::get(state) *
              SoViewingMatrixElement::get(state);
            m = m.inverse();
            m.multLeft(lm);
            PRIVATE(this)->bumprender->renderBumpSpecular(state, PRIVATE(this)->pvcache,
                                                          (SoLight*) lights[i], m);
          }
        }

      }


      PRIVATE(this)->unlock();

      glPopAttrib();
      // we used two units in the bumpmap code
      SoGLMultiTextureImageElement::restore(state, 0);
      SoGLMultiTextureImageElement::restore(state, 1);
      SoGLLazyElement::getInstance(state)->reset(state,
                                                 SoLazyElement::LIGHT_MODEL_MASK|
                                                 SoLazyElement::BLENDING_MASK);

      return FALSE;
    }
  }


  if (shapestyleflags & SoShapeStyleElement::VERTEXARRAY) {
    // lock since pvcache is shared among all threads
    PRIVATE(this)->lock();
    this->validatePVCache(action);
    PRIVATE(this)->unlock();

    SoGLCacheContextElement::shouldAutoCache(state,
                                             SoGLCacheContextElement::DONT_AUTO_CACHE);
    int arrays = SoPrimitiveVertexCache::NORMAL|SoPrimitiveVertexCache::COLOR;
    SoGLMultiTextureImageElement::Model model;
    SbColor blendcolor;
    SoGLImage * glimage = SoGLMultiTextureImageElement::get(state, 0, model, blendcolor);
    if (glimage) arrays |= SoPrimitiveVertexCache::TEXCOORD;
    SoMaterialBundle mb(action);
    mb.sendFirst();
    PRIVATE(this)->setupShapeHints(this, state);
    PRIVATE(this)->pvcache->renderTriangles(state, arrays);
    if (PRIVATE(this)->pvcache->getNumLineIndices() ||
        PRIVATE(this)->pvcache->getNumPointIndices()) {
      const SoNormalElement * nelem = SoNormalElement::getInstance(state);
      if (nelem->getNum() == 0) {
        glPushAttrib(GL_LIGHTING_BIT);
        glDisable(GL_LIGHTING);
        arrays &= SoPrimitiveVertexCache::NORMAL;
      }
      PRIVATE(this)->pvcache->renderLines(state, arrays);
      PRIVATE(this)->pvcache->renderPoints(state, arrays);

      if (nelem->getNum() == 0) {
        glPopAttrib();
      }
    }
    // we have rendered, return FALSE
    return FALSE;
  }

#if COIN_DEBUG && 0 // enable this to test generatePrimitives() rendering
  SoMaterialBundle mb(action);
  mb.sendFirst();
  soshape_get_staticdata()->currentbundle = &mb;  // needed in the primitive callbacks
  this->generatePrimitives(action);
  return FALSE;
#else // generatePrimitives() rendering
  if (PRIVATE(this)->rendercnt < ((1<<SoShapeP::RENDERCNT_BITS)-1)) {
    PRIVATE(this)->rendercnt++;
  }
  return TRUE; // let the shape node render the geometry using OpenGL
#endif // ! generatePrimitives() rendering
}
#endif

/*!
  \COININTERNAL
*/
SbBool
SoShape::shouldRayPick(SoRayPickAction * const action)
{
  switch (SoPickStyleElement::get(action->getState())) {
  case SoPickStyleElement::SHAPE:
  case SoPickStyleElement::SHAPE_ON_TOP:
  case SoPickStyleElement::SHAPE_FRONTFACES:
    return TRUE;
  case SoPickStyleElement::BOUNDING_BOX:
  case SoPickStyleElement::BOUNDING_BOX_ON_TOP:
    this->rayPickBoundingBox(action);
    return FALSE;
  case SoPickStyleElement::UNPICKABLE:
    return FALSE;
  default:
    assert(0 && "unknown pick style");
    return TRUE;
  }
}

#if COIN_BUILD_LEGACY_GL_RENDERER
/*!
  \COININTERNAL
 */
void
SoShape::beginSolidShape(SoGLRenderAction * action)
{
  SoState * state = action->getState();
  state->push();

  SoShapeHintsElement::set(state,
                           SoShapeHintsElement::COUNTERCLOCKWISE,
                           SoShapeHintsElement::SOLID,
                           SoShapeHintsElement::FACE_TYPE_AS_IS);
}

/*!
  \COININTERNAL
*/
void
SoShape::endSolidShape(SoGLRenderAction * action)
{
  action->getState()->pop();
}
#endif

/*!
  \COININTERNAL
*/
void
SoShape::computeObjectSpaceRay(SoRayPickAction * const action)
{
  action->setObjectSpace();
}

/*!
  \COININTERNAL
*/
void
SoShape::computeObjectSpaceRay(SoRayPickAction * const action,
                               const SbMatrix & matrix)
{
  action->setObjectSpace(matrix);
}

/*!
  Will create triangle detail for a SoPickedPoint. This method will
  only be called internally, when generatePrimitives() is used for
  picking (SoShape::rayPick() is not overridden).

  This method returns \c NULL in Open Inventor, and subclasses will
  need to override this method to create details for a SoPickedPoint.

  This is not necessary with Coin. Of course, if you choose to
  override it, it will work in the same way as Open Inventor.

  For this to work, you must supply a face or line detail when
  generating primitives. If you supply \c NULL for the detail argument in
  SoShape::beginShape(), you'll have to override this method.
*/
SoDetail *
SoShape::createTriangleDetail(SoRayPickAction * COIN_UNUSED_ARG(action),
                              const SoPrimitiveVertex * COIN_UNUSED_ARG(v1),
                              const SoPrimitiveVertex * COIN_UNUSED_ARG(v2),
                              const SoPrimitiveVertex * COIN_UNUSED_ARG(v3),
                              SoPickedPoint * COIN_UNUSED_ARG(pp))
{
  soshape_staticdata * shapedata = soshape_get_staticdata();

  if (shapedata->primdata->faceDetail) {
    return shapedata->primdata->createPickDetail();
  }
  // don't warn here. SoDetail instances are optional for extension nodes.
#if COIN_DEBUG && 0
  SoDebugError::postInfo("SoShape::createTriangleDetail",
                         "Unable to create triangle detail.");
#endif // COIN_DEBUG
  return NULL;
}

/*!
  Will create line detail for a SoPickedPoint. This method will only
  be called internally, when generatePrimitives() is used for picking
  (SoShape::rayPick() is not overridden).

  This method returns \c NULL in Open Inventor, and subclasses will
  need to override this method to create details for a SoPickedPoint.

  This is not necessary with Coin. Of course, if you choose to
  override it, it will work in the same way as Open Inventor.

  For this to work, you must supply a face or line detail when
  generating primitives. If you supply \c NULL for the detail argument in
  SoShape::beginShape(), you'll have to override this method.
*/
SoDetail *
SoShape::createLineSegmentDetail(SoRayPickAction * COIN_UNUSED_ARG(action),
                                 const SoPrimitiveVertex * COIN_UNUSED_ARG(v1),
                                 const SoPrimitiveVertex * COIN_UNUSED_ARG(v2),
                                 SoPickedPoint * COIN_UNUSED_ARG(pp))
{
  soshape_staticdata * shapedata = soshape_get_staticdata();

  if (shapedata->primdata->lineDetail) {
    return shapedata->primdata->createPickDetail();
  }
  // don't warn here. SoDetail instances are optional for extension nodes.
#if COIN_DEBUG && 0
  SoDebugError::postInfo("SoShape::createLineSegmentDetail",
                         "Unable to create line segment detail.");
#endif // COIN_DEBUG
  return NULL;
}

/*!
  Will create point detail for a SoPickedPoint. This method will only
  be called internally, when generatePrimitives() is used for picking
  (SoShape::rayPick() is not overridden).

  This method returns \c NULL in Open Inventor, and subclasses will
  need to override this method to create details for a SoPickedPoint.

  This is not necessary with Coin. Of course, if you choose to
  override it, it will work in the same way as Open Inventor.

  For this to work, you must supply a point detail in the
  SoPrimitiveVertex in generatePrimitives().
*/
SoDetail *
SoShape::createPointDetail(SoRayPickAction * /* action */,
                           const SoPrimitiveVertex * v,
                           SoPickedPoint * /* pp */)
{
  if (v->getDetail()) return v->getDetail()->copy();
  return NULL;
}

/*!
  \COININTERNAL
*/
void
SoShape::invokeTriangleCallbacks(SoAction * const action,
                                 const SoPrimitiveVertex * const v1,
                                 const SoPrimitiveVertex * const v2,
                                 const SoPrimitiveVertex * const v3)
{
  if (action->getTypeId().isDerivedFrom(SoRayPickAction::getClassTypeId())) {
    SoRayPickAction * ra = (SoRayPickAction *) action;

    SbVec3f intersection;
    SbVec3f barycentric;
    SbBool front;

    if (ra->intersect(v1->getPoint(), v2->getPoint(), v3->getPoint(),
                      intersection, barycentric, front)) {

      if (ra->isBetweenPlanes(intersection)) {
        if (SoShapeHintsElement::getVertexOrdering(ra->getState()) ==
            SoShapeHintsElement::CLOCKWISE) {
          front = !front;
        }
        SoPickedPoint * pp = ra->addIntersection(intersection, front);
        if (pp) {
          pp->setDetail(this->createTriangleDetail(ra, v1, v2, v3, pp), this);
          // calculate normal at picked point
          SbVec3f n =
            v1->getNormal() * barycentric[0] +
            v2->getNormal() * barycentric[1] +
            v3->getNormal() * barycentric[2];
          n.normalize();
          pp->setObjectNormal(n);

          // calculate texture coordinate at picked point
          SbVec4f tc =
            v1->getTextureCoords() * barycentric[0] +
            v2->getTextureCoords() * barycentric[1] +
            v3->getTextureCoords() * barycentric[2];

          pp->setObjectTextureCoords(tc);

          // material index need to be approximated, since there is no
          // way to average material indices :( This makes it
          // impossible to fully support color per vertex. An
          // extension to the OIV API would perhaps be a good idea
          // here? Maybe calculate the rgba value for diffuse and
          // transparency and set it in SoPickedPoint?
          float maxval = barycentric[0];
          const SoPrimitiveVertex * maxv = v1;
          if (barycentric[1] > maxval) {
            maxv = v2;
            maxval = barycentric[1];
          }
          if (barycentric[2] > maxval) {
            maxv = v3;
          }
          pp->setMaterialIndex(maxv->getMaterialIndex());
        }
      }
    }
  }
  else if (action->getTypeId().isDerivedFrom(SoCallbackAction::getClassTypeId())) {
    SoCallbackAction * ca = (SoCallbackAction *) action;
    ca->invokeTriangleCallbacks(this, v1, v2, v3);
  }
  else if (action->getTypeId().isDerivedFrom(SoGetPrimitiveCountAction::getClassTypeId())) {
    SoGetPrimitiveCountAction * ga = (SoGetPrimitiveCountAction *) action;
    ga->incNumTriangles();
  }
  else if (action->getTypeId().isDerivedFrom(SoIRRenderAction::getClassTypeId())) {
    SoIRRenderAction * ir = static_cast<SoIRRenderAction *>(action);
    SoIRRenderAction::PrimitiveCollector * collector =
      ir->getActivePrimitiveCollector();
    if (collector) collector->onTriangle(v1, v2, v3);
  }
#if COIN_BUILD_LEGACY_GL_RENDERER
  else if (action->getTypeId().isDerivedFrom(SoGLRenderAction::getClassTypeId())) {
    soshape_staticdata * shapedata = soshape_get_staticdata();

    switch (shapedata->rendermode) {
    case SORTED_TRIANGLES:
      shapedata->trianglesort->triangle(action->getState(), v1, v2, v3);
      break;
    case BIGTEXTURE:
      shapedata->currentbigtexture->triangle(action->getState(), v1, v2, v3);
      break;
    case PVCACHE:
      {
        int pdidx[3];
        pdidx[0] = shapedata->primdata->getPointDetailIndex(v1);
        pdidx[1] = shapedata->primdata->getPointDetailIndex(v2);
        pdidx[2] = shapedata->primdata->getPointDetailIndex(v3);
        PRIVATE(this)->pvcache->addTriangle(v1, v2, v3, pdidx);
      }
      break;
    default:
      glBegin(GL_TRIANGLES);
      glTexCoord4fv(v1->getTextureCoords().getValue());
      glNormal3fv(v1->getNormal().getValue());
      shapedata->currentbundle->send(v1->getMaterialIndex(), TRUE);
      glVertex3fv(v1->getPoint().getValue());

      glTexCoord4fv(v2->getTextureCoords().getValue());
      glNormal3fv(v2->getNormal().getValue());
      shapedata->currentbundle->send(v2->getMaterialIndex(), TRUE);
      glVertex3fv(v2->getPoint().getValue());

      glTexCoord4fv(v3->getTextureCoords().getValue());
      glNormal3fv(v3->getNormal().getValue());
      shapedata->currentbundle->send(v3->getMaterialIndex(), TRUE);
      glVertex3fv(v3->getPoint().getValue());
      glEnd();
      break;
    }
  }
#endif
}

/*!
  \COININTERNAL
*/
void
SoShape::invokeLineSegmentCallbacks(SoAction * const action,
                                    const SoPrimitiveVertex * const v1,
                                    const SoPrimitiveVertex * const v2)
{
  if (action->getTypeId().isDerivedFrom(SoRayPickAction::getClassTypeId())) {
    SoRayPickAction * ra = (SoRayPickAction *) action;

    SbVec3f intersection;
    if (ra->intersect(v1->getPoint(), v2->getPoint(), intersection)) {
      if (ra->isBetweenPlanes(intersection)) {
        SoPickedPoint * pp = ra->addIntersection(intersection);
        if (pp) {
          pp->setDetail(this->createLineSegmentDetail(ra, v1, v2, pp), this);
          float total = (v2->getPoint()-v1->getPoint()).length();
          float len1 = 1.0f;
          float len2 = 0.0f;
          if (total > 0.0f) {
            len1 = (intersection-v1->getPoint()).length();
            len2 = (intersection-v2->getPoint()).length();
            len1 /= total;
            len2 /= total;
          }
          SbVec3f n =
            v1->getNormal() * len1 +
            v2->getNormal() * len2;
          n.normalize();
          pp->setObjectNormal(n);

          SbVec4f tc =
            v1->getTextureCoords() * len1 +
            v2->getTextureCoords() * len2;
          pp->setObjectTextureCoords(tc);
          pp->setMaterialIndex(len1 >= len2 ?
                               v1->getMaterialIndex() :
                               v2->getMaterialIndex());

        }
      }
    }
  }
  else if (action->getTypeId().isDerivedFrom(SoCallbackAction::getClassTypeId())) {
    SoCallbackAction * ca = (SoCallbackAction *) action;
    ca->invokeLineSegmentCallbacks(this, v1, v2);
  }
  else if (action->getTypeId().isDerivedFrom(SoGetPrimitiveCountAction::getClassTypeId())) {
    SoGetPrimitiveCountAction * ga = (SoGetPrimitiveCountAction *) action;
    ga->incNumLines();
  }
  else if (action->getTypeId().isDerivedFrom(SoIRRenderAction::getClassTypeId())) {
    SoIRRenderAction * ir = static_cast<SoIRRenderAction *>(action);
    SoIRRenderAction::PrimitiveCollector * collector =
      ir->getActivePrimitiveCollector();
    if (collector) collector->onLine(v1, v2);
  }
#if COIN_BUILD_LEGACY_GL_RENDERER
  else if (action->getTypeId().isDerivedFrom(SoGLRenderAction::getClassTypeId())) {
    soshape_staticdata * shapedata = soshape_get_staticdata();
    switch (shapedata->rendermode) {
    case PVCACHE:
      PRIVATE(this)->pvcache->addLine(v1, v2);
      break;
    default:
      glBegin(GL_LINES);
      glTexCoord4fv(v1->getTextureCoords().getValue());
      glNormal3fv(v1->getNormal().getValue());
      shapedata->currentbundle->send(v1->getMaterialIndex(), TRUE);
      glVertex3fv(v1->getPoint().getValue());

      glTexCoord4fv(v2->getTextureCoords().getValue());
      glNormal3fv(v2->getNormal().getValue());
      shapedata->currentbundle->send(v2->getMaterialIndex(), TRUE);
      glVertex3fv(v2->getPoint().getValue());
      glEnd();
      break;
    }
  }
#endif
}

/*!
  \COININTERNAL
*/
void
SoShape::invokePointCallbacks(SoAction * const action,
                              const SoPrimitiveVertex * const v)
{
  if (action->getTypeId().isDerivedFrom(SoRayPickAction::getClassTypeId())) {
    SoRayPickAction * ra = (SoRayPickAction *) action;

    SbVec3f intersection = v->getPoint();
    if (ra->intersect(intersection)) {
      if (ra->isBetweenPlanes(intersection)) {
        SoPickedPoint * pp = ra->addIntersection(intersection);
        if (pp) {
          pp->setDetail(this->createPointDetail(ra, v, pp), this);
          pp->setObjectNormal(v->getNormal());
          pp->setObjectTextureCoords(v->getTextureCoords());
          pp->setMaterialIndex(v->getMaterialIndex());
        }
      }
    }
  }
  else if (action->getTypeId().isDerivedFrom(SoCallbackAction::getClassTypeId())) {
    SoCallbackAction * ca = (SoCallbackAction *) action;
    ca->invokePointCallbacks(this, v);
  }
  else if (action->getTypeId().isDerivedFrom(SoGetPrimitiveCountAction::getClassTypeId())) {
    SoGetPrimitiveCountAction * ga = (SoGetPrimitiveCountAction *) action;
    ga->incNumPoints();
  }
  else if (action->getTypeId().isDerivedFrom(SoIRRenderAction::getClassTypeId())) {
    SoIRRenderAction * ir = static_cast<SoIRRenderAction *>(action);
    SoIRRenderAction::PrimitiveCollector * collector =
      ir->getActivePrimitiveCollector();
    if (collector) collector->onPoint(v);
  }
#if COIN_BUILD_LEGACY_GL_RENDERER
  else if (action->getTypeId().isDerivedFrom(SoGLRenderAction::getClassTypeId())) {
    soshape_staticdata * shapedata = soshape_get_staticdata();

    switch (shapedata->rendermode) {
    case PVCACHE:
      PRIVATE(this)->pvcache->addPoint(v);
      break;
    default:
      glBegin(GL_POINTS);
      glTexCoord4fv(v->getTextureCoords().getValue());
      glNormal3fv(v->getNormal().getValue());
      shapedata->currentbundle->send(v->getMaterialIndex(), TRUE);
      glVertex3fv(v->getPoint().getValue());
      glEnd();
      break;
    }
  }
#endif
}

/*!

  This method is used to generate primitives for a shape. It's
  typically called from a node's generatePrimitives() method. If you
  have your own shape and want to write a generatePrimitives() method
  for that shape, it's probably a good idea to take a peek in the
  generatePrimitives() method for a similar shape in Coin.

  generatePrimitives() can contain several beginShape()/endShape()
  sequences. shapeVertex() is used for each vertex between
  beginShape() and endShape(). For instance, to generate primitives
  for a triangle you'd do something like this:

  \verbatim
  SoPrimitiveVertex vertex;

  this->beginShape(action, SoShape::POLYGON);
  vertex.setPoint(SbVec3f(0.0f, 0.0f, 0.0f));
  this->shapeVertex(&vertex);
  vertex.setPoint(SbVec3f(1.0f, 0.0f, 0.0f));
  this->shapeVertex(&vertex);
  vertex.setPoint(SbVec3f(1.0f, 1.0f, 0.0f));
  this->shapeVertex(&vertex);
  this->endShape();
  \endverbatim

  Note that the SoPrimitiveVertex instance can simply be placed on the
  stack and not allocated. SoShape will copy the needed information
  when you call shapeVertex().

  Before calling shapeVertex(), you can set extra information for the
  SoPrimitiveVertex, including normal, material index, and texture
  coordinates.

  This method is slightly different from its counterpart from the
  original Open Inventor library, as this method has an SoDetail as
  the last argument, and not an SoFaceDetail. This is because we
  accept more TriangleShape types, and the detail might be a
  SoFaceDetail or a SoLineDetail. There is no use sending in a
  SoPointDetail, as nothing will be done with it.
*/
void
SoShape::beginShape(SoAction * const action, const TriangleShape shapetype,
                    SoDetail * const detail)
{
  soshape_get_staticdata()->primdata->beginShape(this, action, shapetype, detail);
}


/*!

  This method is used while generating primitives for a shape. See
  beginShape() for more details.

  \sa beginShape(), endShape()
*/
void
SoShape::shapeVertex(const SoPrimitiveVertex * const v)
{
  soshape_get_staticdata()->primdata->shapeVertex(v);
}

/*!

  This method is used while generating primitives for a shape. See
  beginShape() for more details.

  \sa beginShape(), shapeVertex()
*/
void
SoShape::endShape(void)
{
  soshape_get_staticdata()->primdata->endShape();
}

/*!
  Convenience function which sets up an SoPrimitiveVertex, and sends
  it using the SoShape::shapeVertex() function. 2D version
*/
void
SoShape::generateVertex(SoPrimitiveVertex * const pv,
                        const SbVec3f & point,
                        const SbBool usetexfunc,
                        const SoMultiTextureCoordinateElement * const tce,
                        const float s,
                        const float t,
                        const SbVec3f & normal)
{
  this->generateVertex(pv, point, usetexfunc, tce, s, t, 0.0f, normal);
}

/*!
  Convenience function which sets up an SoPrimitiveVertex, and sends
  it using the SoShape::shapeVertex() function. 3D version.

  \COIN_FUNCTION_EXTENSION

  \since Coin 2.0
*/
void
SoShape::generateVertex(SoPrimitiveVertex * const pv,
                        const SbVec3f & point,
                        const SbBool usetexfunc,
                        const SoMultiTextureCoordinateElement * const tce,
                        const float s,
                        const float t,
                        const float r,
                        const SbVec3f & normal)
{
  SbVec4f texCoord;
  if (usetexfunc)
    texCoord = tce->get(point, normal);
  else
    texCoord.setValue(s, t, r, 1.0f);
  pv->setPoint(point);
  pv->setNormal(normal);
  pv->setTextureCoords(texCoord);
  shapeVertex(pv);
}

// Doc in superclass.
SbBool
SoShape::affectsState(void) const
{
  // Overridden from default setting in SoNode to return FALSE instead
  // of TRUE, as we know for certain that no node classes derived from
  // SoShape will affect the rendering state.
  return FALSE;
}

// Doc in superclass.
void
SoShape::getPrimitiveCount(SoGetPrimitiveCountAction * action)
{
  if (this->shouldPrimitiveCount(action)) this->generatePrimitives(action);
}

/*!
  Not implemented in Coin. Should probably have been private in TGS
  Inventor API.
 */
float
SoShape::getDecimatedComplexity(SoState * COIN_UNUSED_ARG(state), float complexity)
{
  COIN_OBSOLETED();
  return 1.0f * complexity;
}

/*!
  Render a bounding box.
*/
#if COIN_BUILD_LEGACY_GL_RENDERER
void
SoShape::GLRenderBoundingBox(SoGLRenderAction * action)
{
  SbBox3f box;
  SbVec3f center;
  this->getBBox(action, box, center);
  center = (box.getMin() + box.getMax()) * 0.5f;
  SbVec3f size = box.getMax()  - box.getMin();

  SoMaterialBundle mb(action);
  mb.sendFirst();

  {
    SoGLShapeHintsElement::forceSend(action->getState(), TRUE, FALSE, FALSE);
  }

  glPushMatrix();
  glTranslatef(center[0], center[1], center[2]);
  sogl_render_cube(size[0], size[1], size[2], &mb,
                   SOGL_NEED_NORMALS | SOGL_NEED_TEXCOORDS, NULL);
  glPopMatrix();
}
#endif

/*!
  \COININTERNAL
 */
SbBool
SoShape::shouldPrimitiveCount(SoGetPrimitiveCountAction * COIN_UNUSED_ARG(action))
{
  return TRUE; // FIXME: what to do here? pederb 1999-11-25
}

//
// used when pickStyle == BOUNDING_BOX
//
void
SoShape::rayPickBoundingBox(SoRayPickAction * action)
{
  SbBox3f box;
  SbVec3f center;
  this->getBBox(action, box, center);
  if (box.isEmpty()) return;
  this->computeObjectSpaceRay(action);
  SbVec3f isect;
  if (action->intersect(box, isect, FALSE)) {
    if (action->isBetweenPlanes(isect)) {
      action->addIntersection(isect);
    }
  }
}

// Doc from superclass.
void
SoShape::notify(SoNotList * nl)
{
  inherited::notify(nl);
  PRIVATE(this)->lock();
  if (PRIVATE(this)->bboxcache) {
    PRIVATE(this)->bboxcache->invalidate();
  }
#if COIN_BUILD_LEGACY_GL_RENDERER
  if (PRIVATE(this)->pvcache) {
    PRIVATE(this)->pvcache->invalidate();
  }
#endif
  PRIVATE(this)->flags &= ~SoShapeP::SHOULD_BBOX_CACHE;
  PRIVATE(this)->rendercnt = 0;
  PRIVATE(this)->irCacheValid = false;
  PRIVATE(this)->irRuns.clear();
  PRIVATE(this)->unlock();
}

/*!
  Return the bounding box cache for this shape. It might return
  NULL if no bounding box cache has been created. If not NULL, the
  caller must check if the cache is valid before using it. This
  can be done using SoCache::isValid().

  \COIN_FUNCTION_EXTENSION

  \since Coin 2.0
*/
const SoBoundingBoxCache *
SoShape::getBoundingBoxCache(void) const
{
  return PRIVATE(this)->bboxcache;
}

// return the bbox for this shape, using the cache if valid,
// calculating it if not.
void
SoShape::getBBox(SoAction * action, SbBox3f & box, SbVec3f & center)
{
  SoState * state = action->getState();
  SbBool isvalid = PRIVATE(this)->bboxcache && PRIVATE(this)->bboxcache->isValid(state);
  if (isvalid) {
    box = PRIVATE(this)->bboxcache->getProjectedBox();
    // we know center will be set, so just fetch it from the cache
    center = PRIVATE(this)->bboxcache->getCenter();

    return;
  }

  // destroy the old cache if we have one
  if (PRIVATE(this)->bboxcache) {
    PRIVATE(this)->lock();
    PRIVATE(this)->bboxcache->unref();
    PRIVATE(this)->bboxcache = NULL;
    PRIVATE(this)->unlock();
    // don't create bbox caches for shapes that change
    PRIVATE(this)->flags &= ~SoShapeP::SHOULD_BBOX_CACHE;
  }

  SbBool shouldcache = (PRIVATE(this)->flags & SoShapeP::SHOULD_BBOX_CACHE) != 0;
  SbBool storedinvalid = FALSE;
  if (shouldcache) {
    // must push state to make cache dependencies work
    state->push();
    storedinvalid = SoCacheElement::setInvalid(FALSE);
    assert(PRIVATE(this)->bboxcache == NULL);
    PRIVATE(this)->lock();
    PRIVATE(this)->bboxcache = new SoBoundingBoxCache(state);
    PRIVATE(this)->bboxcache->ref();
    PRIVATE(this)->unlock();
    SoCacheElement::set(state, PRIVATE(this)->bboxcache);
  }
  SbTime begin = SbTime::getTimeOfDay();
  this->computeBBox(action, box, center);
  SbTime end = SbTime::getTimeOfDay();
  if (shouldcache) {
    PRIVATE(this)->bboxcache->set(box, TRUE, center);
    // pop state since we pushed it
    state->pop();
    SoCacheElement::setInvalid(storedinvalid);
  }
  // only create cache if calculating it took longer than the limit
  else if ((end.getValue() - begin.getValue()) >= SoShapeP::bboxcachetimelimit) {
    PRIVATE(this)->flags |= SoShapeP::SHOULD_BBOX_CACHE;
    if (action->isOfType(SoGetBoundingBoxAction::getClassTypeId())) {
      // just recalculate the bbox so that the cache is created at
      // once. SoGLRenderAction and SoRayPickAction might need it.
      state->push();
      storedinvalid = SoCacheElement::setInvalid(FALSE);
      assert(PRIVATE(this)->bboxcache == NULL);
      PRIVATE(this)->lock();
      PRIVATE(this)->bboxcache = new SoBoundingBoxCache(state);
      PRIVATE(this)->bboxcache->ref();
      PRIVATE(this)->unlock();
      SoCacheElement::set(state, PRIVATE(this)->bboxcache);
      box.makeEmpty();
      this->computeBBox(action, box, center);
      PRIVATE(this)->bboxcache->set(box, TRUE, center);
      // pop state since we pushed it
      state->pop();
      SoCacheElement::setInvalid(storedinvalid);
    }
  }
}

void
SoShapeP::calibrateBBoxCache(void)
{
  int i;
  const int ARRAYSIZE = 100;

  // just create 100 random vertices
  SbVec3f vecarray[ARRAYSIZE];
  for (i = 0; i < ARRAYSIZE; i++) {
    for (int j = 0; j < 3; j++) {
      vecarray[i][j] = ((float) rand()) / ((float) RAND_MAX);
    }
  }

  // FIXME: should really measure CPU time spent, and not just wall
  // time. See the item in Coin/docs/todo.txt on implementing a
  // "stopwatch" ADT. 20021111 mortene.
  SbTime begin = SbTime::getTimeOfDay();
  SbBox3f bbox;
  bbox.makeEmpty();
  for (i = 0; i < ARRAYSIZE; i++) {
    bbox.extendBy(vecarray[i]);
  }
  SbTime end = SbTime::getTimeOfDay();
  SoShapeP::bboxcachetimelimit = end.getValue() - begin.getValue();
}

/*!
  Convenience method that enables vertex arrays and/or VBOs
  Returns \e TRUE if VBO is used.

  \sa finishVertexArray()
  \since Coin 3.0
*/
#if COIN_BUILD_LEGACY_GL_RENDERER
SbBool
SoShape::startVertexArray(SoGLRenderAction * action,
                          const SoCoordinateElement * coords,
                          const SbVec3f * pervertexnormals,
                          const SbBool texpervertex,
                          const SbBool colorpervertex)
{
  SoState * state = action->getState();
  const cc_glglue * glue = sogl_glue_instance(state);
  const SoGLVBOElement * vboelem = SoGLVBOElement::getInstance(state);
  const uint32_t contextid = action->getCacheContext();

  SbBool dovbo = TRUE;
  if (!SoGLDriverDatabase::isSupported(glue, SO_GL_VBO_IN_DISPLAYLIST)) {
    if (SoCacheElement::anyOpen(state)) {
      dovbo = FALSE;
    }
  }
  SoVBO * vertexvbo = dovbo ? vboelem->getVertexVBO() : NULL;
  if (!vertexvbo) dovbo = FALSE;
  SbBool didbind = FALSE;

  if (colorpervertex) {
    const GLvoid * dataptr = NULL;
    SoVBO * colorvbo = dovbo ? vboelem->getColorVBO() : NULL;
    SoGLLazyElement * lelem = (SoGLLazyElement*) SoLazyElement::getInstance(state);
    if (colorvbo) {
      lelem->updateColorVBO(colorvbo);
      colorvbo->bindBuffer(contextid);
      didbind = TRUE;
    }
    else {
      if (didbind) {
        cc_glglue_glBindBuffer(glue, GL_ARRAY_BUFFER, 0);
        didbind = FALSE;
      }
      dataptr = (const GLvoid*) lelem->getDiffusePointer();
    }
    if (colorvbo) {
      cc_glglue_glColorPointer(glue, 4, GL_UNSIGNED_BYTE, 0, dataptr);
    }
    else {
      cc_glglue_glColorPointer(glue, 3, GL_FLOAT, 0, dataptr);
    }
    cc_glglue_glEnableClientState(glue, GL_COLOR_ARRAY);
  }
  if (texpervertex) {
    const SoMultiTextureCoordinateElement * mtelem = NULL;
    const SbBool * enabledunits = NULL;
    int lastenabled;
    
    enabledunits = SoMultiTextureEnabledElement::getEnabledUnits(state, lastenabled);
    if (enabledunits) {
      mtelem = SoMultiTextureCoordinateElement::getInstance(state);
    }
    SoVBO * vbo;
    if (!SoGLDriverDatabase::isSupported(glue, SO_GL_MULTITEXTURE)) {
      static int hasWarned = 0;
      if (lastenabled>0) {
	if (!hasWarned) {
	  SoDebugError::postWarning("SoShape::startVertexArray",
				    "Multitexturing is not supported on this hardware, but more than one textureunit is in use."
				  );
	  hasWarned = 1;
	}
      }
      lastenabled = 0;
    }

    for (int i = 0; i <= lastenabled; i++) {
      if (enabledunits[i] && mtelem->getNum(i)) {
        int dim = mtelem->getDimension(i);
        const GLvoid * tptr;
        switch (dim) {
        default:
        case 2: tptr = (const GLvoid*) mtelem->getArrayPtr2(i); break;
        case 3: tptr = (const GLvoid*) mtelem->getArrayPtr3(i); break;
        case 4: tptr = (const GLvoid*) mtelem->getArrayPtr4(i); break;
        }
	if (SoGLDriverDatabase::isSupported(glue, SO_GL_MULTITEXTURE)) {
	  cc_glglue_glClientActiveTexture(glue, GL_TEXTURE0 + i);
	}
        vbo = dovbo ? vboelem->getTexCoordVBO(i) : NULL;
        if (vbo) {
          vbo->bindBuffer(contextid);
          didbind = TRUE;
          tptr = NULL;
        }
        else {
          if (didbind) {
            cc_glglue_glBindBuffer(glue, GL_ARRAY_BUFFER, 0);
            didbind = FALSE;
          }
        }
        cc_glglue_glTexCoordPointer(glue, dim, GL_FLOAT, 0, tptr);
        cc_glglue_glEnableClientState(glue, GL_TEXTURE_COORD_ARRAY);
      }
    }
  }
  if (pervertexnormals != NULL) {
    SoVBO * vbo = dovbo ? vboelem->getNormalVBO() : NULL;
    const GLvoid * dataptr = NULL;
    if (vbo) {
      vbo->bindBuffer(contextid);
      didbind = TRUE;
    }
    else {
      dataptr = (const GLvoid*) pervertexnormals;
      if (didbind) {
        cc_glglue_glBindBuffer(glue, GL_ARRAY_BUFFER, 0);
        didbind = FALSE;
      }
    }
    cc_glglue_glNormalPointer(glue, GL_FLOAT, 0, dataptr);
    cc_glglue_glEnableClientState(glue, GL_NORMAL_ARRAY);
  }
  const GLvoid * dataptr = NULL;
  if (vertexvbo) {
    vertexvbo->bindBuffer(contextid);
  }
  else {
    dataptr = coords->is3D() ?
      ((const GLvoid *)coords->getArrayPtr3()) :
      ((const GLvoid *)coords->getArrayPtr4());
  }
  cc_glglue_glVertexPointer(glue, coords->is3D() ? 3 : 4, GL_FLOAT, 0,
                            dataptr);
  cc_glglue_glEnableClientState(glue, GL_VERTEX_ARRAY);

  SoGLVertexAttributeElement::getInstance(state)->enableVBO(action);

  return dovbo;
}

/*!
  Should be called after rendering with vertex arrays. This method
  will disable arrays and VBOs enabled in the startVertexArray()
  function.

  \sa startVertexArray()
  \since Coin 3.0
*/
void
SoShape::finishVertexArray(SoGLRenderAction * action,
                           const SbBool vbo,
                           const SbBool normpervertex,
                           const SbBool texpervertex,
                           const SbBool colorpervertex)
{
  SoState * state = action->getState();
  const cc_glglue * glue = sogl_glue_instance(state);

  if (vbo) {
    if (!SoGLDriverDatabase::isSupported(glue, SO_GL_VBO_IN_DISPLAYLIST)) {
      SoCacheElement::invalidate(state);
      SoGLCacheContextElement::shouldAutoCache(state,
                                               SoGLCacheContextElement::DONT_AUTO_CACHE);
    }
    // unset VBO buffer
    cc_glglue_glBindBuffer(glue, GL_ARRAY_BUFFER, 0);
  }
  cc_glglue_glDisableClientState(glue, GL_VERTEX_ARRAY);
  if (normpervertex) {
    cc_glglue_glDisableClientState(glue, GL_NORMAL_ARRAY);
  }
  if (texpervertex) {
    int lastenabled;
    const SbBool * enabledunits =
      SoMultiTextureEnabledElement::getEnabledUnits(state, lastenabled);
    if (!SoGLDriverDatabase::isSupported(glue, SO_GL_MULTITEXTURE)) {
      //Should already have warned in StartVertexArray
      lastenabled = 0;
    }
    
    const SoMultiTextureCoordinateElement * mtelem =
      SoMultiTextureCoordinateElement::getInstance(state);
    
    for (int i = 0; i <= lastenabled; i++) {
      if (enabledunits[i] && mtelem->getNum(i)) {
	if (SoGLDriverDatabase::isSupported(glue, SO_GL_MULTITEXTURE)) {
	  cc_glglue_glClientActiveTexture(glue, GL_TEXTURE0 + i);
	}
        cc_glglue_glDisableClientState(glue, GL_TEXTURE_COORD_ARRAY);
      }
    }
    cc_glglue_glClientActiveTexture(glue, GL_TEXTURE0);
  }
  if (colorpervertex) {
    SoGLLazyElement * lelem = (SoGLLazyElement*) SoLazyElement::getInstance(state);
    lelem->reset(state, SoLazyElement::DIFFUSE_MASK);

    cc_glglue_glDisableClientState(glue, GL_COLOR_ARRAY);
  }

  SoGLVertexAttributeElement::getInstance(state)->disableVBO(action);
}
#endif

#if COIN_BUILD_LEGACY_GL_RENDERER
void
SoShape::validatePVCache(SoGLRenderAction * action)
{
  SoState * state = action->getState();
  if (PRIVATE(this)->pvcache == NULL ||
      !PRIVATE(this)->pvcache->isValid(state)) {
    if (PRIVATE(this)->pvcache) {
      PRIVATE(this)->pvcache->unref();
    }
    // we don't want to create display list caches while building the VBOs
    SoCacheElement::invalidate(state);

    soshape_staticdata * shapedata = soshape_get_staticdata();
    SbBool storedinvalid = SoCacheElement::setInvalid(FALSE);
    // must push state to make cache dependencies work
    state->push();
    PRIVATE(this)->pvcache = new SoPrimitiveVertexCache(state);
    PRIVATE(this)->pvcache->ref();
    SoCacheElement::set(state, PRIVATE(this)->pvcache);
    shapedata->rendermode = PVCACHE;
    this->generatePrimitives(action);
    shapedata->rendermode = NORMAL;
    // needed for out old bumpmap handling
    if (PRIVATE(this)->bumprender) PRIVATE(this)->bumprender->calcTangentSpace(PRIVATE(this)->pvcache);
    // this _must_ be called after creating the pvcache

    // FIXME: consider if we should call a virtual function here to
    // enable subclasses to modify the primitive vertex cache. Must be
    // done before to state->pop() call.
    state->pop();
    SoCacheElement::setInvalid(storedinvalid);
    PRIVATE(this)->pvcache->close(state);
    PRIVATE(this)->testSetupShapeHints(this);
  }
}
#endif


#undef PRIVATE
