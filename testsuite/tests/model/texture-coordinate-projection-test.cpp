// testsuite/texture-coordinate-projection-test.cpp
//
// Host (no GPU) contract tests for SoTextureCoordinateProjection's effect on
// the retained render IR.  The node installs a texture-coordinate *function*
// through SoMultiTextureCoordinateElement; during SoIRRenderAction traversal
// SoShape materialises the generated coordinates into the command's
// geometry.texcoords stream.  These tests lock:
//   * the four mappings produce the documented object-space projections,
//   * scale and offset are applied as u = p * scale + offset,
//   * BOX picks the dominant-axis plane per face (checked on each box axis),
//   * the node overrides an inherited SoTextureCoordinate2 only within its
//     scope and the inherited coordinates are restored afterwards.
//
// A texture has to be enabled for the coordinate bundle to request texture
// coordinates at all, so every scene carries a 1x1 SoTexture2.

#include <Inventor/SoDB.h>
#include <Inventor/SbViewportRegion.h>
#include <Inventor/actions/SoIRRenderAction.h>
#include <Inventor/nodes/SoCoordinate3.h>
#include <Inventor/nodes/SoIndexedFaceSet.h>
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/nodes/SoTexture2.h>
#include <Inventor/nodes/SoTextureCoordinate2.h>
#include <Inventor/nodes/SoTextureCoordinateProjection.h>

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void
check(bool condition, const std::string & message)
{
  if (!condition) {
    std::cerr << "FAIL: " << message << std::endl;
    ++failures;
  }
}

void
checkNear(float actual, float expected, const std::string & message)
{
  if (std::fabs(actual - expected) > 1.0e-4f) {
    std::ostringstream os;
    os << message << " (actual=" << actual << ", expected=" << expected << ")";
    check(false, os.str());
  }
}

// Flattened command geometry copied out of the IR so it stays valid after the
// action and the scene are gone.
struct Geometry {
  uint32_t vertexCount = 0;
  std::vector<float> positions;  // 3 per vertex
  std::vector<float> normals;    // 3 per vertex (may be empty)
  std::vector<float> texcoords;  // 4 per vertex (may be empty)
};

Geometry
readGeometry(SoNode * root)
{
  SoIRRenderAction action(SbViewportRegion(64, 64));
  action.apply(root);

  Geometry geometry;
  const SoDrawList & drawlist = action.getDrawList();
  for (int c = 0; c < drawlist.getNumCommands(); ++c) {
    const SoGeometryDesc & desc = drawlist.getCommand(c).geometry;
    const uint32_t posStride = desc.vertexStride
      ? desc.vertexStride / sizeof(float) : 3u;
    const uint32_t texStride = desc.texcoordStride
      ? desc.texcoordStride / sizeof(float) : 4u;
    for (uint32_t i = 0; i < desc.vertexCount; ++i) {
      for (uint32_t k = 0; k < 3; ++k) {
        geometry.positions.push_back(desc.positions[i * posStride + k]);
      }
      if (desc.normals) {
        for (uint32_t k = 0; k < 3; ++k) {
          geometry.normals.push_back(desc.normals[i * 3 + k]);
        }
      }
      if (desc.texcoords) {
        for (uint32_t k = 0; k < 4; ++k) {
          geometry.texcoords.push_back(desc.texcoords[i * texStride + k]);
        }
      }
    }
    geometry.vertexCount += desc.vertexCount;
  }
  return geometry;
}

// A 1x1 white texture, needed so the coordinate bundle materialises
// coordinates at all.
SoTexture2 *
makeTexture()
{
  SoTexture2 * texture = new SoTexture2;
  unsigned char white = 255;
  texture->image.setValue(SbVec2s(1, 1), 1, &white);
  return texture;
}

// A self-contained quad: the coordinate node and the indexed face set kept
// together in one separator so the face set always has its coordinates.
SoSeparator *
makeQuad(const std::vector<SbVec3f> & corners)
{
  SoSeparator * quadNode = new SoSeparator;
  SoCoordinate3 * coordinates = new SoCoordinate3;
  for (size_t i = 0; i < corners.size(); ++i) {
    coordinates->point.set1Value(static_cast<int>(i), corners[i]);
  }
  quadNode->addChild(coordinates);
  SoIndexedFaceSet * faces = new SoIndexedFaceSet;
  faces->coordIndex.setNum(static_cast<int>(corners.size()) + 1);
  for (size_t i = 0; i < corners.size(); ++i) {
    faces->coordIndex.set1Value(static_cast<int>(i),
                                static_cast<int32_t>(i));
  }
  faces->coordIndex.set1Value(static_cast<int>(corners.size()), -1);
  quadNode->addChild(faces);
  return quadNode;
}

// One quad under the given projection.  corners must be in the coordinate
// node's index order; makeQuad() reuses that order.
Geometry
projectQuad(const std::vector<SbVec3f> & corners,
            SoTextureCoordinateProjection::Mapping mapping,
            const SbVec3f & scale = SbVec3f(1.0f, 1.0f, 1.0f),
            const SbVec3f & offset = SbVec3f(0.0f, 0.0f, 0.0f))
{
  SoSeparator * root = new SoSeparator;
  root->ref();
  root->addChild(makeTexture());

  SoCoordinate3 * coordinates = new SoCoordinate3;
  for (size_t i = 0; i < corners.size(); ++i) {
    coordinates->point.set1Value(static_cast<int>(i), corners[i]);
  }
  root->addChild(coordinates);

  SoTextureCoordinateProjection * projection = new SoTextureCoordinateProjection;
  projection->mapping = mapping;
  projection->scale.setValue(scale);
  projection->offset.setValue(offset);
  root->addChild(projection);

  SoIndexedFaceSet * faces = new SoIndexedFaceSet;
  faces->coordIndex.setNum(static_cast<int>(corners.size()) + 1);
  for (size_t i = 0; i < corners.size(); ++i) {
    faces->coordIndex.set1Value(static_cast<int>(i),
                                static_cast<int32_t>(i));
  }
  faces->coordIndex.set1Value(static_cast<int>(corners.size()), -1);
  root->addChild(faces);

  const Geometry geometry = readGeometry(root);
  root->unref();
  return geometry;
}

float
uv(const Geometry & geometry, uint32_t vertex, uint32_t component)
{
  return geometry.texcoords[static_cast<size_t>(vertex) * 4 + component];
}

}  // namespace

int
main()
{
  SoDB::init();

  // A quad in the z=0 plane, normal +Z.
  const std::vector<SbVec3f> quad = {
    SbVec3f(0.0f, 0.0f, 0.0f),
    SbVec3f(2.0f, 0.0f, 0.0f),
    SbVec3f(2.0f, 1.0f, 0.0f),
    SbVec3f(0.0f, 1.0f, 0.0f)
  };

  // --- The four mappings generate coordinates ----------------------------
  const Geometry planar =
    projectQuad(quad, SoTextureCoordinateProjection::PLANAR);
  const Geometry box = projectQuad(quad, SoTextureCoordinateProjection::BOX);
  const Geometry spherical =
    projectQuad(quad, SoTextureCoordinateProjection::SPHERICAL);
  const Geometry cylindrical =
    projectQuad(quad, SoTextureCoordinateProjection::CYLINDRICAL);

  check(!planar.texcoords.empty(), "PLANAR produced no texture coordinates");
  check(!spherical.texcoords.empty(),
        "SPHERICAL produced no texture coordinates");
  check(!cylindrical.texcoords.empty(),
        "CYLINDRICAL produced no texture coordinates");

  if (!planar.texcoords.empty() && !spherical.texcoords.empty() &&
      !cylindrical.texcoords.empty()) {
    check(planar.texcoords != spherical.texcoords,
          "SPHERICAL must differ from PLANAR");
    check(planar.texcoords != cylindrical.texcoords,
          "CYLINDRICAL must differ from PLANAR");
    check(spherical.texcoords != cylindrical.texcoords,
          "SPHERICAL must differ from CYLINDRICAL");
  }

  // For a +Z face the BOX projection degenerates to the planar (x,y) plane.
  check(planar.texcoords == box.texcoords,
        "BOX on a +Z face must match PLANAR");

  // PLANAR is u = x, v = y (offset 0, unit scale).
  if (planar.vertexCount >= 4) {
    for (uint32_t i = 0; i < planar.vertexCount; ++i) {
      checkNear(uv(planar, i, 0), planar.positions[i * 3 + 0],
                "PLANAR u == x");
      checkNear(uv(planar, i, 1), planar.positions[i * 3 + 1],
                "PLANAR v == y");
    }
  }

  // --- scale and offset are applied --------------------------------------
  {
    const Geometry scaled = projectQuad(
      quad, SoTextureCoordinateProjection::PLANAR,
      SbVec3f(2.0f, 3.0f, 1.0f), SbVec3f(0.25f, 0.5f, 0.0f));
    check(scaled.vertexCount == planar.vertexCount,
          "scale/offset changed the vertex count");
    if (scaled.vertexCount == planar.vertexCount &&
        scaled.vertexCount >= 4) {
      for (uint32_t i = 0; i < scaled.vertexCount; ++i) {
        checkNear(uv(scaled, i, 0),
                  scaled.positions[i * 3 + 0] * 2.0f + 0.25f,
                  "PLANAR u applies scale and offset");
        checkNear(uv(scaled, i, 1),
                  scaled.positions[i * 3 + 1] * 3.0f + 0.5f,
                  "PLANAR v applies scale and offset");
      }
    }
  }

  // --- BOX chooses the dominant axis per face ----------------------------
  {
    // Normal +X face (x is constant): BOX uses (y, z).
    const std::vector<SbVec3f> xFace = {
      SbVec3f(1.0f, -1.0f, -1.0f),
      SbVec3f(1.0f,  1.0f, -1.0f),
      SbVec3f(1.0f,  1.0f,  1.0f),
      SbVec3f(1.0f, -1.0f,  1.0f)
    };
    const Geometry boxX =
      projectQuad(xFace, SoTextureCoordinateProjection::BOX);
    for (uint32_t i = 0; i < boxX.vertexCount; ++i) {
      checkNear(uv(boxX, i, 0), boxX.positions[i * 3 + 1],
                "BOX +X face u == y");
      checkNear(uv(boxX, i, 1), boxX.positions[i * 3 + 2],
                "BOX +X face v == z");
    }

    // Normal +Y face (y is constant): BOX uses (x, z).
    const std::vector<SbVec3f> yFace = {
      SbVec3f(-1.0f, 1.0f, -1.0f),
      SbVec3f( 1.0f, 1.0f, -1.0f),
      SbVec3f( 1.0f, 1.0f,  1.0f),
      SbVec3f(-1.0f, 1.0f,  1.0f)
    };
    const Geometry boxY =
      projectQuad(yFace, SoTextureCoordinateProjection::BOX);
    for (uint32_t i = 0; i < boxY.vertexCount; ++i) {
      checkNear(uv(boxY, i, 0), boxY.positions[i * 3 + 0],
                "BOX +Y face u == x");
      checkNear(uv(boxY, i, 1), boxY.positions[i * 3 + 2],
                "BOX +Y face v == z");
    }
  }

  // --- Override an inherited SoTextureCoordinate2, then restore ----------
  {
    SoSeparator * root = new SoSeparator;
    root->ref();
    root->addChild(makeTexture());

    SoTextureCoordinate2 * inherited = new SoTextureCoordinate2;
    inherited->point.set1Value(0, SbVec2f(0.0f, 0.0f));
    inherited->point.set1Value(1, SbVec2f(0.1f, 0.0f));
    inherited->point.set1Value(2, SbVec2f(0.1f, 0.1f));
    inherited->point.set1Value(3, SbVec2f(0.0f, 0.1f));
    root->addChild(inherited);

    // First quad uses the inherited explicit coordinates.
    root->addChild(makeQuad(quad));

    // Scoped override: only the second quad sees the projection.
    SoSeparator * scope = new SoSeparator;
    scope->addChild(makeTexture());
    SoTextureCoordinateProjection * projection =
      new SoTextureCoordinateProjection;
    projection->mapping = SoTextureCoordinateProjection::PLANAR;
    scope->addChild(projection);
    scope->addChild(makeQuad(quad));
    root->addChild(scope);

    // Third quad must fall back to the inherited explicit coordinates.
    root->addChild(makeQuad(quad));

    const Geometry geometry = readGeometry(root);
    root->unref();

    check(geometry.vertexCount == 3u * 6u,
          "override scene: expected three triangulated quads");
    if (geometry.vertexCount == 18 && !geometry.texcoords.empty()) {
      // Inherited explicit coords (default PER_VERTEX_INDEXED binding) for the
      // first quad.
      checkNear(uv(geometry, 0, 0), 0.0f, "inherited u before the scope");
      checkNear(uv(geometry, 1, 0), 0.1f, "inherited u before the scope");
      // The scoped projection maps to object x/y: vertex 6 is the first
      // vertex of the middle quad, at (x, y) = (0, 0).
      checkNear(uv(geometry, 6, 0), 0.0f, "projection u inside the scope");
      checkNear(uv(geometry, 6, 1), 0.0f, "projection v inside the scope");
      // The last quad is restored to the inherited explicit coords.
      checkNear(uv(geometry, 12, 0), 0.0f, "inherited u after the scope");
      checkNear(uv(geometry, 13, 0), 0.1f, "inherited u after the scope");
    }
  }

  SoDB::finish();

  if (failures != 0) {
    std::cerr << failures << " check(s) failed" << std::endl;
  }
  return failures == 0 ? 0 : 1;
}
