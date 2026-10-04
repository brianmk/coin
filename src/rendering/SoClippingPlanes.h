#ifndef COIN_SOCLIPPINGPLANES_H
#define COIN_SOCLIPPINGPLANES_H

/*!
  \file SoClippingPlanes.h
  \brief Shared GL/Vulkan camera clipping-plane computation.

  SoRenderManagerP (GL) and SoVulkanRenderManagerP compute near/far from the
  scene bounding box with the same algorithm; this shared core keeps the two
  paths from diverging.
*/

#include <Inventor/SbBox3f.h>

#include <cmath>
#include <limits>

//! Slack factor applied by both callers when writing the computed planes.
static const float kSoClippingSlack = 0.001f;

/*!
  \brief Compute near/far from a camera-space projected box.

  Clipping offset is 1% of the box diagonal (clamped to [epsilon, 1.0]); an
  empty box defaults to near=1, far=10.  For perspective cameras the near
  plane is the FIXED_NEAR_PLANE value or the VARIABLE_NEAR_PLANE precision
  limit.

  \a autoClipping uses the managers' enum values: 0=NO_AUTO_CLIPPING,
  1=FIXED_NEAR_PLANE, 2=VARIABLE_NEAR_PLANE.

  \return FALSE when the caller must keep its current planes (whole scene
  behind a non-orthographic camera); on TRUE \a nearval/\a farval hold the
  pre-slack planes, to which the caller applies kSoClippingSlack.
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
