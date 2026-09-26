#ifndef COIN_SOPHYSICALMATERIAL_H
#define COIN_SOPHYSICALMATERIAL_H

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
#include <Inventor/fields/SoMFBool.h>
#include <Inventor/fields/SoMFFloat.h>
#include <Inventor/fields/SoSFFloat.h>

class COIN_DLL_API SoPhysicalMaterial : public SoNode {
  typedef SoNode inherited;

  SO_NODE_HEADER(SoPhysicalMaterial);

public:
  static void initClass(void);
  SoPhysicalMaterial(void);

  //! Metalness of the metallic-roughness model: 0 = dielectric
  //! (plastic/concrete), 1 = conductor (metal).  May hold one value (applied
  //! to every face) or one value per per-face material index.
  SoMFFloat metalness;
  //! Perceptual roughness: 0 = mirror, 1 = fully diffuse.  May hold one value
  //! (applied to every face) or one value per per-face material index.
  SoMFFloat roughness;
  //! When FALSE (or when the node is absent) the shape keeps the legacy
  //! Blinn-Phong appearance and the two fields above are ignored.  May hold
  //! one value (applied to every face) or one value per per-face material
  //! index.
  SoMFBool enabled;
  //! Scalars applied to the optional roughness, normal and emissive texture
  //! maps.  Default 1 (as authored).
  SoSFFloat roughnessStrength;
  SoSFFloat normalStrength;
  SoSFFloat emissiveIntensity;

  //! Index of refraction of the dielectric (glass) response used when the
  //! material is transmissive (its opacity/transparency < 1).  Default 1.5
  //! (window glass).  An enabled physical material authors its own optics, so
  //! the viewport's global glass default applies only to materials without
  //! one.
  SoSFFloat transmissionIor;
  //! Beer-Lambert absorption strength of the dielectric response: 0 is a
  //! perfectly clear glass, higher values tint/thicken the transmission with
  //! distance through the medium.  Default 0.
  SoSFFloat transmissionAbsorption;

  void doAction(SoAction * action) override;
  void callback(SoCallbackAction * action) override;

protected:
  virtual ~SoPhysicalMaterial();
};

#endif // !COIN_SOPHYSICALMATERIAL_H
