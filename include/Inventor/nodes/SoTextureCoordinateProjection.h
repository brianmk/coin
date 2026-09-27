#ifndef COIN_SOTEXTURECOORDINATEPROJECTION_H
#define COIN_SOTEXTURECOORDINATEPROJECTION_H

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

#include <Inventor/nodes/SoSubNode.h>
#include <Inventor/nodes/SoTextureCoordinateFunction.h>
#include <Inventor/fields/SoSFEnum.h>
#include <Inventor/fields/SoSFVec3f.h>
#include <Inventor/SbVec4f.h>
#include <Inventor/SbVec3f.h>

class COIN_DLL_API SoTextureCoordinateProjection : public SoTextureCoordinateFunction {
  typedef SoTextureCoordinateFunction inherited;

  SO_NODE_HEADER(SoTextureCoordinateProjection);

public:
  //! How object coordinates are turned into texture coordinates.
  enum Mapping {
    //! Project the object X/Y coordinates onto the texture (planar).
    PLANAR,
    //! Pick the object-coordinate plane most aligned with the surface
    //! normal, per triangle, so every face of a box gets a sensible mapping.
    BOX,
    //! Wrap by spherical coordinates.
    SPHERICAL,
    //! Wrap by cylindrical coordinates around the object Z axis.
    CYLINDRICAL
  };

  static void initClass(void);
  SoTextureCoordinateProjection(void);

  //! Projection mode. Defaults to PLANAR.
  SoSFEnum mapping;
  //! Multiplier applied to the projected coordinates: tiles per object unit.
  //! Defaults to (1, 1, 1).
  SoSFVec3f scale;
  //! Constant offset added to the resulting texture coordinates, in UV units.
  //! Defaults to (0, 0, 0).
  SoSFVec3f offset;

  void doAction(SoAction * action) override;
  void pick(SoPickAction * action) override;
  void callback(SoCallbackAction * action) override;

protected:
  virtual ~SoTextureCoordinateProjection();

private:
  static const SbVec4f & generate(void * userdata,
                                  const SbVec3f & p,
                                  const SbVec3f & n);

  SbVec4f dummy_projection;
};

#endif // !COIN_SOTEXTURECOORDINATEPROJECTION_H
