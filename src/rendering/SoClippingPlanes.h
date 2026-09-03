// src/rendering/SoClippingPlanes.h

#ifndef COIN_SOCLIPPINGPLANES_H
#define COIN_SOCLIPPINGPLANES_H

/*!
  \file SoClippingPlanes.h
  \brief Shared camera clipping-plane computation for the scene managers.

  The legacy OpenGL SoRenderManagerP and the Vulkan SoVulkanRenderManagerP
  compute the camera near/far planes from the scene bounding box with the same
  algorithm.  This header holds the shared pure-math core so GL-side fixes
  cannot silently diverge from the Vulkan path (and vice versa).
*/

#include <Inventor/SbBox3f.h>

#include <cmath>
#include <limits>

//! Slack factor applied by both callers when writing the computed planes.
static const float kSoClippingSlack = 0.001f;

/*!
  \brief Compute near/far clipping planes from a camera-space projected box.

  Pure function of the bounding box and the clipping strategy:

  - clipping offset of 1% of the box diagonal (at most 1.0, at least epsilon),
  - empty-box defaults (near=1, far=10),
  - the perspective near-plane precision limit (VARIABLE_NEAR_PLANE) or the
    fixed near-plane value (FIXED_NEAR_PLANE).

  \a autoClipping uses the numeric values of the two managers' enums:
  0 = NO_AUTO_CLIPPING, 1 = FIXED_NEAR_PLANE, 2 = VARIABLE_NEAR_PLANE.

  \return FALSE when the caller must keep its current planes (the
  whole-scene-behind-the-camera case for non-orthographic cameras); on TRUE
  \a nearval/\a farval hold the pre-slack planes.  The caller applies
  kSoClippingSlack and writes the result to its own state.
*/
inline bool
coinComputeClippingPlanes(const SbBox3f & box,
                          const bool isOrthographic,
                          const bool isPerspective,
                          const int autoClipping,
                          const float nearplanevalue,
                          float & nearval,
                          float & farval)
{
  float sizeX, sizeY, sizeZ;
  box.getSize(sizeX, sizeY, sizeZ);
  const float boxDiagonal =
    std::sqrt(sizeX * sizeX + sizeY * sizeY + sizeZ * sizeZ);

  // Clipping offset is 1% of the bounding box diagonal, at most 1.0 and at
  // least std::numeric_limits<float>::epsilon().
  const float clippingOffset =
    SbMin(1.0f, SbMax(std::numeric_limits<float>::epsilon(),
                      0.01f * boxDiagonal));
  nearval = -box.getMax()[2] - clippingOffset;
  farval = -box.getMin()[2] + clippingOffset;

  if (!isOrthographic && farval <= 0.0f) {
    return false;
  }

  if (box.isEmpty()) {
    nearval = 1;
    farval = 10;
  }

  if (isPerspective) {
    float nearlimit;
    if (autoClipping == 1) { // FIXED_NEAR_PLANE
      nearlimit = nearplanevalue;
    }
    else {
      const int depthbits = 32;
      const int use_bits = static_cast<int>(
        static_cast<float>(depthbits) * (1.0f - nearplanevalue));
      const float r = static_cast<float>(
        std::pow(2.0, static_cast<double>(use_bits)));
      nearlimit = farval / r;
    }

    if (nearlimit >= farval) {
      nearlimit = farval / 5000.0f;
    }

    if (nearval < nearlimit) {
      nearval = nearlimit;
    }
  }
  return true;
}

#endif // COIN_SOCLIPPINGPLANES_H
