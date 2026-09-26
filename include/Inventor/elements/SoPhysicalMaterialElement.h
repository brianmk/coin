#ifndef COIN_SOPHYSICALMATERIALELEMENT_H
#define COIN_SOPHYSICALMATERIALELEMENT_H

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

#include <Inventor/elements/SoReplacedElement.h>

#include <vector>

class COIN_DLL_API SoPhysicalMaterialElement : public SoReplacedElement {
  typedef SoReplacedElement inherited;

  SO_ELEMENT_HEADER(SoPhysicalMaterialElement);
public:
  static void initClass(void);
protected:
  virtual ~SoPhysicalMaterialElement();

public:
  void init(SoState * state) override;

  //! Carry an authored metallic-roughness material through the traversal
  //! state.  All three arrays are indexed by the primitive's material index
  //! (the same index SoLazyElement uses for per-face colors); a single-element
  //! array applies to every face.  An empty \a enabled array means the element
  //! is inactive and the legacy Blinn-Phong appearance is used.
  static void set(SoState * const state, SoNode * const node,
                  const std::vector<unsigned char> & enabled,
                  const std::vector<float> & metalness,
                  const std::vector<float> & roughness,
                  const float roughnessStrength,
                  const float normalStrength,
                  const float emissiveIntensity,
                  const float transmissionIor,
                  const float transmissionAbsorption);
  static void get(SoState * const state, SbBool & enabled,
                  float & metalness, float & roughness);
  static void get(SoState * const state, SbBool & enabled,
                  float & metalness, float & roughness,
                  const int materialIndex);
  //! Scalar map strengths shared by every face of the material.
  static void getStrengths(SoState * const state,
                           float & roughnessStrength,
                           float & normalStrength,
                           float & emissiveIntensity);
  //! Dielectric optics shared by every face: index of refraction and
  //! Beer-Lambert absorption strength.
  static void getOptics(SoState * const state, float & transmissionIor,
                        float & transmissionAbsorption);

  SbBool matches(const SoElement * element) const override;
  SoElement * copyMatchInfo(void) const override;

  void print(FILE * file) const override;

protected:
  std::vector<unsigned char> enabled;
  std::vector<float> metalness;
  std::vector<float> roughness;
  float roughnessStrength;
  float normalStrength;
  float emissiveIntensity;
  float transmissionIor;
  float transmissionAbsorption;
};

#endif // !COIN_SOPHYSICALMATERIALELEMENT_H
