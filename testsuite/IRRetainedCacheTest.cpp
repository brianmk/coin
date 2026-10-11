/**************************************************************************\
 * Regression tests for the retained-IR render cache (Vulkan-only stale
 * geometry bug).
 *
 * Unlike the in-tree `#ifdef COIN_TEST_SUITE` blocks this file is a plain
 * testsuite source: it is compiled straight into the `CoinTests` executable
 * (see testsuite/CMakeLists.txt) and only uses the public API.
 *
 * Background
 * ----------
 * `SoShape::IRRender()` caches its tessellation keyed on the shape's own
 * fields plus SoComplexity, but the vertex data is read from *inherited*
 * traversal state (`SoCoordinate3` / `SoVertexProperty` / `SoNormal` /
 * `SoTextureCoordinate2`).  FreeCAD's Sketcher rewrites a sibling
 * `SoCoordinate3` on every drag step, which never notifies the shape, so the
 * Vulkan retained path replayed stale geometry forever while GL re-read state
 * each frame ("dragging a sketch freezes geometry on Vulkan").
 *
 * Fix A: the cache also keys on the coordinate/vertex-property source node id
 * and num, and snapshots the normal / multi-texture-coordinate elements'
 * `copyMatchInfo()` to compare them with `matches()`.
 *
 * Fix B: every retained rebuild publishes a monotonic
 * `SoGeometryDesc::retainedGeneration`; the Vulkan backend compares it in
 * `identityMatches()` in addition to the raw pointers.  Without it a rebuild
 * that gets the just-freed streams' address back from the allocator (the
 * common case) looks identical to the backend and the GPU buffer is never
 * re-uploaded (pointer-identity ABA).
 *
 * The sibling-coordinate part of Fix A already has coverage in
 * `src/shapenodes/SoShape.cpp` (extracted to `shapenodesSoShapeTest.cpp`).
 * These tests extend it: the Sketcher point-marker configuration
 * (`SoMarkerSet` + `SoMaterialBinding::PER_VERTEX`), over-invalidation
 * protection for material-only edits, generation advancement on the
 * normal/texcoord paths, and the ABA consumer-key scenario.
\**************************************************************************/

#include "CoinTest.h"

#include <Inventor/SbVec2s.h>
#include <Inventor/SbViewportRegion.h>
#include <Inventor/actions/SoIRRenderAction.h>
#include <Inventor/nodes/SoCoordinate3.h>
#include <Inventor/nodes/SoIndexedLineSet.h>
#include <Inventor/nodes/SoLineSet.h>
#include <Inventor/nodes/SoMarkerSet.h>
#include <Inventor/nodes/SoMaterial.h>
#include <Inventor/nodes/SoMaterialBinding.h>
#include <Inventor/nodes/SoNormal.h>
#include <Inventor/nodes/SoNormalBinding.h>
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/nodes/SoTexture2.h>
#include <Inventor/nodes/SoTextureCoordinate2.h>
#include <Inventor/nodes/SoVertexProperty.h>
#include <Inventor/rendering/SoRenderIR.h>

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

// Everything a regression assertion may need from one recorded command.
struct IRProbe {
  bool has = false;
  SoPrimitiveTopology topology = SO_TOPOLOGY_TRIANGLES;
  const float * positions = nullptr;
  uint32_t vertexCount = 0;
  uint32_t normalCount = 0;
  uint64_t generation = 0;
  bool retained = false;
  std::vector<float> pos;
  std::vector<float> nrm;
  std::vector<float> tex;
  std::vector<float> colors;
};

// Probe the first command of `action` whose topology matches (the default);
// pass a different topology to select a command out of a mixed draw list.
IRProbe
probe_command(SoIRRenderAction & action, SoPrimitiveTopology topology)
{
  IRProbe probe;
  const SoDrawList & list = action.getDrawList();
  for (int i = 0; i < list.getNumCommands(); ++i) {
    const SoGeometryDesc & geometry = list.getCommand(i).geometry;
    if (geometry.topology != topology) continue;
    probe.has = true;
    probe.topology = geometry.topology;
    probe.positions = geometry.positions;
    probe.vertexCount = geometry.vertexCount;
    probe.normalCount = geometry.normalCount;
    probe.generation = geometry.retainedGeneration;
    probe.retained = geometry.retained;
    if (geometry.positions && geometry.vertexCount) {
      probe.pos.assign(geometry.positions,
                       geometry.positions + 3 * geometry.vertexCount);
    }
    if (geometry.normals && geometry.normalCount) {
      probe.nrm.assign(geometry.normals,
                       geometry.normals + 3 * geometry.normalCount);
    }
    if (geometry.texcoords && geometry.vertexCount) {
      const uint32_t stride =
        (geometry.texcoordStride ? geometry.texcoordStride : 16u) / 4u;
      probe.tex.assign(geometry.texcoords,
                       geometry.texcoords + stride * geometry.vertexCount);
    }
    if (geometry.colors && geometry.vertexCount) {
      probe.colors.assign(geometry.colors,
                          geometry.colors + 4 * geometry.vertexCount);
    }
    return probe;
  }
  return probe;
}

IRProbe
probe_lines(SoIRRenderAction & action)
{
  return probe_command(action, SO_TOPOLOGY_LINES);
}

// The consumer identity key a backend uses to decide whether an existing GPU
// buffer may be reused.  Mirrors SoVulkanRenderBackend's `identityMatches`
// after Fix B: pointers and counts alone are not enough, the producer's build
// id has to match as well.
struct GeometryConsumerKey {
  const float * positions = nullptr;
  uint32_t vertexCount = 0;
  uint32_t normalCount = 0;
  uint64_t generation = 0;

  static GeometryConsumerKey from(const IRProbe & probe)
  {
    GeometryConsumerKey key;
    key.positions = probe.positions;
    key.vertexCount = probe.vertexCount;
    key.normalCount = probe.normalCount;
    key.generation = probe.generation;
    return key;
  }

  bool matches(const IRProbe & probe) const
  {
    return this->positions == probe.positions &&
      this->vertexCount == probe.vertexCount &&
      this->normalCount == probe.normalCount &&
      this->generation == probe.generation;
  }

  // The pre-fix predicate: retained streams were assumed to change pointer
  // exactly when their content changes, so the content hash was skipped.
  bool legacyPointerOnlyMatches(const IRProbe & probe) const
  {
    return this->positions == probe.positions &&
      this->vertexCount == probe.vertexCount &&
      this->normalCount == probe.normalCount;
  }
};

// A Sketcher-like marker scene: per-vertex material makes the emitted command
// non-retained (the colors come from an in-place arena), but the *positions*
// still come from the shape's retained tessellation cache and must follow a
// sibling SoCoordinate3 rewrite.
SoMarkerSet *
build_marker_set(SoSeparator * root, SoCoordinate3 *& coords, SoMaterial *& material)
{
  coords = new SoCoordinate3;
  coords->point.set1Value(0, 0.0f, 0.0f, 0.0f);
  coords->point.set1Value(1, 1.0f, 0.0f, 0.0f);
  root->addChild(coords);

  auto * binding = new SoMaterialBinding;
  binding->value = SoMaterialBinding::PER_VERTEX;
  root->addChild(binding);

  material = new SoMaterial;
  material->diffuseColor.set1Value(0, 1.0f, 0.0f, 0.0f);
  material->diffuseColor.set1Value(1, 0.0f, 1.0f, 0.0f);
  root->addChild(material);

  auto * markers = new SoMarkerSet;
  markers->numPoints = 2;
  root->addChild(markers);
  return markers;
}

} // namespace

// ---------------------------------------------------------------------------
// A. CPU-only IR coverage: inherited state must invalidate the retained cache.
// ---------------------------------------------------------------------------

// The Sketcher point-marker configuration: a sibling coordinate rewrite must
// re-emit positions even though the command is not `retained` (it carries
// per-vertex colors from the arena).
BOOST_AUTO_TEST_CASE(irRetainedCacheFollowsCoordinatesForPerVertexMarkerSet)
{
  SoSeparator * root = new SoSeparator;
  root->ref();
  SoCoordinate3 * coords = nullptr;
  SoMaterial * material = nullptr;
  build_marker_set(root, coords, material);

  SoIRRenderAction action(SbViewportRegion(800, 600));

  action.apply(root);
  const IRProbe built = probe_command(action, SO_TOPOLOGY_POINTS);
  BOOST_REQUIRE_MESSAGE(built.has, "no point command recorded for SoMarkerSet");
  BOOST_CHECK_EQUAL(built.vertexCount, 2u);
  BOOST_CHECK_MESSAGE(built.pos.size() == 6, "expected two marker positions");
  // Per-vertex colors are re-derived each emission, so the commands are not
  // advertised as shape-retained; the cache still owns the positions.
  BOOST_CHECK_MESSAGE(!built.retained,
                      "per-vertex material command unexpectedly marked retained");
  BOOST_CHECK_EQUAL(built.generation, 0u);

  // Unchanged frame replays the cache.
  action.apply(root);
  const IRProbe idle = probe_command(action, SO_TOPOLOGY_POINTS);
  BOOST_CHECK_MESSAGE(idle.positions == built.positions,
                      "unchanged marker frame replaced the retained streams");

  // Sibling SoCoordinate3 rewrite: never notifies the shape.
  coords->point.set1Value(1, 0.0f, 1.0f, 0.0f);
  action.apply(root);
  const IRProbe moved = probe_command(action, SO_TOPOLOGY_POINTS);
  BOOST_CHECK_MESSAGE(moved.pos != built.pos,
                      "sibling SoCoordinate3 rewrite replayed stale marker positions");
  BOOST_CHECK_MESSAGE(moved.pos.size() == 6 && moved.pos[3] == 0.0f &&
                        moved.pos[4] == 1.0f && moved.pos[5] == 0.0f,
                      "marker positions did not follow the rewritten sibling coordinates");

  root->unref();
}

// A material-only edit must NOT invalidate the retained IR: the colors are
// resolved from the live state on every emission, so keying the cache on them
// would re-tessellate every shape on every colour change.
BOOST_AUTO_TEST_CASE(irRetainedCacheIgnoresMaterialOnlyChanges)
{
  SoSeparator * root = new SoSeparator;
  root->ref();
  SoCoordinate3 * coords = nullptr;
  SoMaterial * material = nullptr;
  build_marker_set(root, coords, material);

  SoIRRenderAction action(SbViewportRegion(800, 600));

  action.apply(root);
  const IRProbe built = probe_command(action, SO_TOPOLOGY_POINTS);
  BOOST_REQUIRE_MESSAGE(built.has, "no point command recorded for SoMarkerSet");
  BOOST_REQUIRE_MESSAGE(built.colors.size() == 8, "expected per-vertex colors");
  BOOST_CHECK_MESSAGE(built.colors[0] == 1.0f && built.colors[1] == 0.0f,
                      "expected the first marker to be red");

  material->diffuseColor.set1Value(0, 0.0f, 0.0f, 1.0f);
  material->diffuseColor.set1Value(1, 1.0f, 1.0f, 0.0f);
  action.apply(root);
  const IRProbe recolored = probe_command(action, SO_TOPOLOGY_POINTS);
  BOOST_REQUIRE_MESSAGE(recolored.has, "no point command after the material edit");
  // The colour arena is rebuilt in place, so only its content is allowed to
  // change -- the retained positions and the build id must survive.
  BOOST_CHECK_MESSAGE(recolored.colors != built.colors ||
                        recolored.colors[1] != built.colors[1],
                      "material-only change did not reach the emitted vertex colors");
  BOOST_CHECK_MESSAGE(recolored.positions == built.positions,
                      "material-only change replaced the retained streams");
  BOOST_CHECK_EQUAL(recolored.generation, built.generation);

  root->unref();
}

// Fix A, normal/texcoord half: rewriting a sibling SoNormal or
// SoTextureCoordinate2 must advance the build id, not merely re-emit content
// through the pointer-identity path.
BOOST_AUTO_TEST_CASE(irRetainedCacheGenerationFollowsNormalAndTexCoordRewrites)
{
  SoSeparator * root = new SoSeparator;
  root->ref();
  auto * coords = new SoCoordinate3;
  coords->point.set1Value(0, 0.0f, 0.0f, 0.0f);
  coords->point.set1Value(1, 1.0f, 0.0f, 0.0f);
  root->addChild(coords);

  auto * nbind = new SoNormalBinding;
  nbind->value = SoNormalBinding::PER_VERTEX_INDEXED;
  root->addChild(nbind);

  auto * normals = new SoNormal;
  normals->vector.set1Value(0, 0.0f, 0.0f, 1.0f);
  normals->vector.set1Value(1, 0.0f, 0.0f, 1.0f);
  root->addChild(normals);

  auto * texture = new SoTexture2;
  unsigned char pixel[3] = { 255, 128, 0 };
  texture->image.setValue(SbVec2s(1, 1), 3, pixel);
  root->addChild(texture);

  auto * texcoords = new SoTextureCoordinate2;
  texcoords->point.set1Value(0, 0.0f, 0.0f);
  texcoords->point.set1Value(1, 0.0f, 0.0f);
  root->addChild(texcoords);

  auto * lines = new SoLineSet;
  lines->numVertices.set1Value(0, 2);
  root->addChild(lines);

  SoIRRenderAction action(SbViewportRegion(800, 600));

  action.apply(root);
  const IRProbe built = probe_lines(action);
  BOOST_REQUIRE_MESSAGE(built.has, "no line command recorded");
  BOOST_REQUIRE_MESSAGE(built.generation != 0, "retained geometry has no build id");

  // Sibling SoNormal rewrite.
  normals->vector.set1Value(0, 0.0f, 1.0f, 0.0f);
  normals->vector.set1Value(1, 0.0f, 1.0f, 0.0f);
  action.apply(root);
  const IRProbe renormalled = probe_lines(action);
  BOOST_CHECK_MESSAGE(renormalled.generation > built.generation,
                      "sibling SoNormal rewrite did not advance retainedGeneration");
  BOOST_CHECK_MESSAGE(renormalled.nrm != built.nrm,
                      "sibling SoNormal rewrite replayed stale normals");

  // Sibling SoTextureCoordinate2 rewrite.
  texcoords->point.set1Value(1, 1.0f, 1.0f);
  action.apply(root);
  const IRProbe retextured = probe_lines(action);
  BOOST_CHECK_MESSAGE(retextured.generation > renormalled.generation,
                      "sibling SoTextureCoordinate2 rewrite did not advance retainedGeneration");
  BOOST_CHECK_MESSAGE(retextured.tex != renormalled.tex,
                      "sibling SoTextureCoordinate2 rewrite replayed stale texcoords");

  // A material-only edit in the same scene must not advance the id.
  auto * material = new SoMaterial;
  material->diffuseColor.setValue(0.5f, 0.5f, 0.5f);
  root->insertChild(material, 0);
  action.apply(root);
  const IRProbe recolored = probe_lines(action);
  BOOST_CHECK_EQUAL(recolored.generation, retextured.generation);

  root->unref();
}

// ---------------------------------------------------------------------------
// B. Generation / pointer-identity ABA.
// ---------------------------------------------------------------------------

// The backend decides between "reuse the uploaded buffer" and "re-upload" from
// the geometry descriptor.  Because a rebuild routinely reallocates the very
// block it just freed, the address can be byte-identical to the previous
// frame's while the content changed; only retainedGeneration distinguishes the
// two.  Assert the invariant unconditionally and report whether the allocator
// actually cooperated by reusing the address.
BOOST_AUTO_TEST_CASE(irRetainedGenerationDefeatsPointerIdentityAba)
{
  SoSeparator * root = new SoSeparator;
  root->ref();
  auto * coords = new SoCoordinate3;
  coords->point.set1Value(0, -1.0f, 0.0f, 0.0f);
  coords->point.set1Value(1, 1.0f, 0.0f, 0.0f);
  root->addChild(coords);

  auto * nbind = new SoNormalBinding;
  nbind->value = SoNormalBinding::PER_VERTEX_INDEXED;
  root->addChild(nbind);
  auto * normals = new SoNormal;
  normals->vector.set1Value(0, 0.0f, 0.0f, 1.0f);
  normals->vector.set1Value(1, 0.0f, 0.0f, 1.0f);
  root->addChild(normals);

  auto * lines = new SoLineSet;
  lines->numVertices.set1Value(0, 2);
  root->addChild(lines);

  SoIRRenderAction action(SbViewportRegion(800, 600));

  action.apply(root);
  IRProbe previous = probe_lines(action);
  BOOST_REQUIRE_MESSAGE(previous.has, "no line command recorded");
  BOOST_REQUIRE_MESSAGE(previous.generation != 0,
                        "retained geometry carries no retainedGeneration");
  BOOST_REQUIRE_MESSAGE(previous.positions != nullptr,
                        "retained line geometry has no positions");

  int rebuilds = 0;
  int addressReuses = 0;
  for (int i = 0; i < 16; ++i) {
    // Force an unconditional rebuild of the retained run (shape notify) *and*
    // move the sibling coordinates it reads, so the new content differs while
    // the allocation size stays identical.
    lines->touch();
    coords->point.set1Value(1, 1.0f, 0.25f * float(i + 1), 0.0f);
    action.apply(root);
    const IRProbe current = probe_lines(action);
    BOOST_REQUIRE_MESSAGE(current.has, "no line command after a rebuild");

    BOOST_CHECK_MESSAGE(current.generation > previous.generation,
                        "rebuild did not advance retainedGeneration");
    BOOST_CHECK_MESSAGE(current.pos != previous.pos,
                        "rebuild replayed the previous line positions");

    const GeometryConsumerKey previousKey = GeometryConsumerKey::from(previous);
    BOOST_CHECK_MESSAGE(!previousKey.matches(current),
                        "consumer keyed on (positions, counts, generation) "
                        "missed an invalidated geometry stream");

    if (current.positions == previous.positions) {
      ++addressReuses;
      // The measured ABA case: same address, different content.  The legacy
      // pointer-only predicate would have kept the stale GPU buffer ...
      BOOST_CHECK_MESSAGE(previousKey.legacyPointerOnlyMatches(current),
                          "expected the legacy pointer-only key to match on an "
                          "address-reusing rebuild (test setup invariant)");
      if (addressReuses == 1) {
        std::fprintf(stderr,
                     "[INFO] pointer-identity ABA reproduced: rebuild %d reused "
                     "position address %p with changed content\n",
                     i, static_cast<const void *>(current.positions));
      }
    }

    previous = current;
    ++rebuilds;
  }

  if (addressReuses == 0) {
    std::fprintf(stderr,
                 "[INFO] allocator did not reuse the freed position address in "
                 "%d rebuilds; the monotonic retainedGeneration invariant was "
                 "still asserted\n",
                 rebuilds);
  }

  root->unref();
}

// The same ABA scenario but driven by a shape *field* write that changes the
// tessellated output while the buffers keep their size: an SoIndexedLineSet's
// coordIndex selects which two of four coordinates are drawn.  The
// coordinates never change, only the shape's own field.
BOOST_AUTO_TEST_CASE(irRetainedGenerationFollowsShapeFieldRewrite)
{
  SoSeparator * root = new SoSeparator;
  root->ref();
  auto * coords = new SoCoordinate3;
  coords->point.set1Value(0, -1.0f, -0.5f, 0.0f);
  coords->point.set1Value(1, 1.0f, -0.5f, 0.0f);
  coords->point.set1Value(2, -1.0f, 0.5f, 0.0f);
  coords->point.set1Value(3, 1.0f, 0.5f, 0.0f);
  root->addChild(coords);

  auto * nbind = new SoNormalBinding;
  nbind->value = SoNormalBinding::PER_VERTEX_INDEXED;
  root->addChild(nbind);
  auto * normals = new SoNormal;
  normals->vector.set1Value(0, 0.0f, 0.0f, 1.0f);
  normals->vector.set1Value(1, 0.0f, 0.0f, 1.0f);
  root->addChild(normals);

  auto * indexed = new SoIndexedLineSet;
  indexed->coordIndex.set1Value(0, 0);
  indexed->coordIndex.set1Value(1, 1);
  indexed->coordIndex.set1Value(2, -1);
  root->addChild(indexed);

  SoIRRenderAction action(SbViewportRegion(800, 600));

  action.apply(root);
  const IRProbe low = probe_lines(action);
  BOOST_REQUIRE_MESSAGE(low.has, "no line command recorded");
  BOOST_REQUIRE_MESSAGE(low.pos.size() == 6, "expected one two-vertex line");
  BOOST_CHECK_MESSAGE(low.pos[1] == -0.5f && low.pos[4] == -0.5f,
                      "expected the first line to sit at y = -0.5");

  // Pure shape-field write: point the indexed set at the other pair.
  indexed->coordIndex.set1Value(0, 2);
  indexed->coordIndex.set1Value(1, 3);
  action.apply(root);
  const IRProbe high = probe_lines(action);
  BOOST_REQUIRE_MESSAGE(high.has, "no line command after the coordIndex rewrite");
  BOOST_CHECK_MESSAGE(high.generation > low.generation,
                      "shape-field rewrite did not advance retainedGeneration");
  BOOST_CHECK_MESSAGE(high.pos != low.pos,
                      "shape-field rewrite replayed stale positions");
  BOOST_CHECK_MESSAGE(high.pos[1] == 0.5f && high.pos[4] == 0.5f,
                      "coordIndex rewrite did not move the emitted line");

  root->unref();
}

// The other vertex source is a sibling SoVertexProperty; it must invalidate the
// retained cache just like SoCoordinate3.
BOOST_AUTO_TEST_CASE(irRetainedCacheFollowsVertexPropertyRewrite)
{
  SoSeparator * root = new SoSeparator;
  root->ref();
  auto * vprop = new SoVertexProperty;
  vprop->vertex.set1Value(0, 0.0f, 0.0f, 0.0f);
  vprop->vertex.set1Value(1, 1.0f, 0.0f, 0.0f);
  root->addChild(vprop);

  auto * lines = new SoLineSet;
  lines->numVertices.set1Value(0, 2);
  root->addChild(lines);

  SoIRRenderAction action(SbViewportRegion(800, 600));

  action.apply(root);
  const IRProbe built = probe_lines(action);
  BOOST_REQUIRE_MESSAGE(built.has, "no line command recorded for SoVertexProperty");
  BOOST_REQUIRE_MESSAGE(built.generation != 0, "retained geometry has no build id");

  vprop->vertex.set1Value(1, 0.0f, 1.0f, 0.0f);
  action.apply(root);
  const IRProbe moved = probe_lines(action);
  BOOST_CHECK_MESSAGE(moved.generation > built.generation,
                      "sibling SoVertexProperty rewrite did not advance retainedGeneration");
  BOOST_CHECK_MESSAGE(moved.pos != built.pos,
                      "sibling SoVertexProperty rewrite replayed stale positions");

  root->unref();
}
