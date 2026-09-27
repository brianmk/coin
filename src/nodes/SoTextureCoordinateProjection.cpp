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
  \class SoTextureCoordinateProjection Inventor/nodes/SoTextureCoordinateProjection.h
  \brief Generates object-space texture coordinates with a selectable projection.

  Unlike SoTextureCoordinateObject (which always projects through fixed
  factors), this node can pick a per-triangle projection from the surface
  normal, so a textured box shows a consistent pattern on every face instead
  of a single smeared stripe on the faces parallel to the projection axis.

  It generates texture coordinates from the object-space position \a p and
  normal \a n passed to the coordinate function, applies \c scale and
  \c offset, and installs the result as the multi-texture coordinate function
  for subsequent shapes.

  <b>FILE FORMAT/DEFAULTS:</b>
  \code
    TextureCoordinateProjection {
        mapping PLANAR
        scale 1 1 1
        offset 0 0 0
    }
  \endcode

  \sa SoTextureCoordinateObject, SoTextureCoordinatePlane
*/

#include <Inventor/actions/SoCallbackAction.h>
#include <Inventor/actions/SoPickAction.h>
#include <Inventor/elements/SoMultiTextureCoordinateElement.h>
#include <Inventor/elements/SoTextureUnitElement.h>
#include <Inventor/nodes/SoTextureCoordinateProjection.h>

#include "nodes/SoSubNodeP.h"

#include <cmath>

// *************************************************************************

SO_NODE_SOURCE(SoTextureCoordinateProjection);

/*!
  \var SoSFEnum SoTextureCoordinateProjection::mapping
  The projection used to turn object coordinates into texture coordinates.
  Defaults to SoTextureCoordinateProjection::PLANAR.
*/
/*!
  \var SoSFVec3f SoTextureCoordinateProjection::scale
  Multiplier applied to the projected object coordinates (before \c offset).
  For PLANAR and BOX this has units of texture tiles per object unit, so a
  value of 1/50 makes one tile span 50 object units. Defaults to (1, 1, 1).
*/
/*!
  \var SoSFVec3f SoTextureCoordinateProjection::offset
  Constant offset added to the generated texture coordinates, in UV units.
  Defaults to (0, 0, 0).
*/

/*!
  Constructor.
*/
SoTextureCoordinateProjection::SoTextureCoordinateProjection(void)
{
  SO_NODE_INTERNAL_CONSTRUCTOR(SoTextureCoordinateProjection);

  SO_NODE_ADD_FIELD(mapping, (PLANAR));
  SO_NODE_ADD_FIELD(scale, (1.0f, 1.0f, 1.0f));
  SO_NODE_ADD_FIELD(offset, (0.0f, 0.0f, 0.0f));

  SO_NODE_DEFINE_ENUM_VALUE(Mapping, PLANAR);
  SO_NODE_DEFINE_ENUM_VALUE(Mapping, BOX);
  SO_NODE_DEFINE_ENUM_VALUE(Mapping, SPHERICAL);
  SO_NODE_DEFINE_ENUM_VALUE(Mapping, CYLINDRICAL);
  SO_NODE_SET_SF_ENUM_TYPE(mapping, Mapping);
}

/*!
  Destructor.
*/
SoTextureCoordinateProjection::~SoTextureCoordinateProjection()
{
}

// doc from parent
/*!
  \copybrief SoBase::initClass(void)
*/
void
SoTextureCoordinateProjection::initClass(void)
{
  SO_NODE_INTERNAL_INIT_CLASS(SoTextureCoordinateProjection, SO_FROM_INVENTOR_2_0);
}

//! Generates a texture coordinate for an object-space point and normal.
const SbVec4f &
SoTextureCoordinateProjection::generate(void * userdata,
                                        const SbVec3f & p,
                                        const SbVec3f & n)
{
  SoTextureCoordinateProjection * thisp =
    static_cast<SoTextureCoordinateProjection *>(userdata);

  const SbVec3f & s = thisp->scale.getValue();
  const SbVec3f & o = thisp->offset.getValue();
  float u = 0.0f;
  float v = 0.0f;

  switch (thisp->mapping.getValue()) {
  case BOX: {
    // Choose the object-coordinate plane whose axis is most aligned with the
    // face normal and use the two remaining coordinates.
    const float ax = std::fabs(n[0]);
    const float ay = std::fabs(n[1]);
    const float az = std::fabs(n[2]);
    if (ax >= ay && ax >= az) {
      u = p[1] * s[1];
      v = p[2] * s[2];
    }
    else if (ay >= ax && ay >= az) {
      u = p[0] * s[0];
      v = p[2] * s[2];
    }
    else {
      u = p[0] * s[0];
      v = p[1] * s[1];
    }
    break;
  }
  case SPHERICAL: {
    const float r = p.length() > 1.0e-9f ? p.length() : 1.0f;
    float z = p[2] / r;
    if (z > 1.0f) z = 1.0f;
    if (z < -1.0f) z = -1.0f;
    u = 0.5f + std::atan2(p[1], p[0]) / (2.0f * static_cast<float>(M_PI));
    v = 0.5f - std::asin(z) / static_cast<float>(M_PI);
    u *= s[0];
    v *= s[1];
    break;
  }
  case CYLINDRICAL: {
    u = 0.5f + std::atan2(p[1], p[0]) / (2.0f * static_cast<float>(M_PI));
    u *= s[0];
    v = p[2] * s[1];
    break;
  }
  case PLANAR:
  default:
    u = p[0] * s[0];
    v = p[1] * s[1];
    break;
  }

  thisp->dummy_projection.setValue(u + o[0], v + o[1], 0.0f, 1.0f);
  return thisp->dummy_projection;
}

// doc from parent
void
SoTextureCoordinateProjection::doAction(SoAction * action)
{
  SoState * state = action->getState();
  int unit = SoTextureUnitElement::get(state);
  SoMultiTextureCoordinateElement::setFunction(action->getState(), this, unit,
                                               SoTextureCoordinateProjection::generate,
                                               this);
}

// doc from parent
void
SoTextureCoordinateProjection::callback(SoCallbackAction * action)
{
  SoTextureCoordinateProjection::doAction(static_cast<SoAction *>(action));
}

// doc from parent
void
SoTextureCoordinateProjection::pick(SoPickAction * action)
{
  SoTextureCoordinateProjection::doAction(static_cast<SoAction *>(action));
}
