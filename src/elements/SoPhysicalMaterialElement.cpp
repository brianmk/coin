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
  \class SoPhysicalMaterialElement Inventor/elements/SoPhysicalMaterialElement.h
  \brief Reserved element for a physically based material.

  Carries the metallic-roughness parameters of the shape currently being
  traversed.  It is unset (enabled == FALSE) by default, which keeps the
  traditional Blinn-Phong appearance; a producer node (for example
  Gui::SoFCPbrMaterial) sets it for materials that carry authored metalness
  and roughness.

  \ingroup coin_elements
*/

#include "SbBasicP.h"

#include <Inventor/elements/SoPhysicalMaterialElement.h>

#include <cassert>
#include <cstdio>

namespace
{
//! Resolve a per-face value from an array that may hold a single value (which
//! applies to every face) or one value per material index.
float physicalValueAt(const std::vector<float> & values,
                      const int materialIndex,
                      const float fallback)
{
  if (values.empty()) {
    return fallback;
  }
  if (values.size() == 1) {
    return values[0];
  }
  int index = materialIndex;
  if (index < 0) {
    index = 0;
  }
  if (index >= static_cast<int>(values.size())) {
    index = static_cast<int>(values.size()) - 1;
  }
  return values[static_cast<size_t>(index)];
}

//! Resolve the per-face "physical material authored" flag.  An empty array
//! means the element is inactive (legacy Blinn-Phong).
bool physicalEnabledAt(const std::vector<unsigned char> & values,
                       const int materialIndex)
{
  if (values.empty()) {
    return false;
  }
  if (values.size() == 1) {
    return values[0] != 0;
  }
  int index = materialIndex;
  if (index < 0) {
    index = 0;
  }
  if (index >= static_cast<int>(values.size())) {
    index = static_cast<int>(values.size()) - 1;
  }
  return values[static_cast<size_t>(index)] != 0;
}
}  // namespace

SO_ELEMENT_SOURCE(SoPhysicalMaterialElement);

/*!
  \copydetails SoElement::initClass(void)
*/

void
SoPhysicalMaterialElement::initClass(void)
{
  SO_ELEMENT_INIT_CLASS(SoPhysicalMaterialElement, inherited);
}

/*!
  Destructor.
*/

SoPhysicalMaterialElement::~SoPhysicalMaterialElement()
{
}

//! FIXME: write doc.

void
SoPhysicalMaterialElement::init(SoState * state)
{
  inherited::init(state);
  this->enabled.clear();
  this->metalness.clear();
  this->roughness.clear();
  this->roughnessStrength = 1.0f;
  this->normalStrength = 1.0f;
  this->emissiveIntensity = 1.0f;
  this->transmissionIor = 1.5f;
  this->transmissionAbsorption = 0.0f;
}

//! FIXME: write doc.

void
SoPhysicalMaterialElement::set(SoState * const state,
                               SoNode * const node,
                               const std::vector<unsigned char> & enabled,
                               const std::vector<float> & metalness,
                               const std::vector<float> & roughness,
                               const float roughnessStrength,
                               const float normalStrength,
                               const float emissiveIntensity,
                               const float transmissionIor,
                               const float transmissionAbsorption)
{
  SoPhysicalMaterialElement * element =
    coin_safe_cast<SoPhysicalMaterialElement *>
    (
     SoReplacedElement::getElement(state, classStackIndex, node)
     );
  if (element) {
    element->enabled = enabled;
    element->metalness = metalness;
    element->roughness = roughness;
    element->roughnessStrength = roughnessStrength;
    element->normalStrength = normalStrength;
    element->emissiveIntensity = emissiveIntensity;
    element->transmissionIor = transmissionIor;
    element->transmissionAbsorption = transmissionAbsorption;
  }
}

//! FIXME: write doc.

void
SoPhysicalMaterialElement::get(SoState * const state, SbBool & enabled,
                               float & metalness, float & roughness)
{
  SoPhysicalMaterialElement::get(state, enabled, metalness, roughness, 0);
}

//! FIXME: write doc.

void
SoPhysicalMaterialElement::get(SoState * const state, SbBool & enabled,
                               float & metalness, float & roughness,
                               const int materialIndex)
{
  const SoPhysicalMaterialElement * element =
    coin_assert_cast<const SoPhysicalMaterialElement *>
    (
     SoElement::getConstElement(state, classStackIndex)
     );
  enabled = physicalEnabledAt(element->enabled, materialIndex);
  metalness = physicalValueAt(element->metalness, materialIndex, 0.0f);
  roughness = physicalValueAt(element->roughness, materialIndex, 0.5f);
}

//! FIXME: write doc.

void
SoPhysicalMaterialElement::getStrengths(SoState * const state,
                                        float & roughnessStrength,
                                        float & normalStrength,
                                        float & emissiveIntensity)
{
  const SoPhysicalMaterialElement * element =
    coin_assert_cast<const SoPhysicalMaterialElement *>
    (
     SoElement::getConstElement(state, classStackIndex)
     );
  roughnessStrength = element->roughnessStrength;
  normalStrength = element->normalStrength;
  emissiveIntensity = element->emissiveIntensity;
}

//! FIXME: write doc.

void
SoPhysicalMaterialElement::getOptics(SoState * const state,
                                     float & transmissionIor,
                                     float & transmissionAbsorption)
{
  const SoPhysicalMaterialElement * element =
    coin_assert_cast<const SoPhysicalMaterialElement *>
    (
     SoElement::getConstElement(state, classStackIndex)
     );
  transmissionIor = element->transmissionIor;
  transmissionAbsorption = element->transmissionAbsorption;
}

//! FIXME: write doc.

SbBool
SoPhysicalMaterialElement::matches(const SoElement * element) const
{
  const SoPhysicalMaterialElement * other =
    coin_assert_cast<const SoPhysicalMaterialElement *>(element);
  return this->enabled == other->enabled &&
    this->metalness == other->metalness &&
    this->roughness == other->roughness &&
    this->roughnessStrength == other->roughnessStrength &&
    this->normalStrength == other->normalStrength &&
    this->emissiveIntensity == other->emissiveIntensity &&
    this->transmissionIor == other->transmissionIor &&
    this->transmissionAbsorption == other->transmissionAbsorption;
}

//! FIXME: write doc.

SoElement *
SoPhysicalMaterialElement::copyMatchInfo(void) const
{
  SoPhysicalMaterialElement * element =
    static_cast<SoPhysicalMaterialElement *>
    (SoPhysicalMaterialElement::getClassTypeId().createInstance());
  element->enabled = this->enabled;
  element->metalness = this->metalness;
  element->roughness = this->roughness;
  element->roughnessStrength = this->roughnessStrength;
  element->normalStrength = this->normalStrength;
  element->emissiveIntensity = this->emissiveIntensity;
  element->transmissionIor = this->transmissionIor;
  element->transmissionAbsorption = this->transmissionAbsorption;
  return element;
}

//! FIXME: write doc.

void
SoPhysicalMaterialElement::print(FILE * file) const
{
  fprintf(file, "SoPhysicalMaterialElement[%p]: enabled[%d] = %s, "
          "metalness[%d] = %f, roughness[%d] = %f, strengths = (%f, %f, %f), "
          "optics = (ior %f, absorption %f)\n",
          this,
          static_cast<int>(this->enabled.size()),
          physicalEnabledAt(this->enabled, 0) ? "true" : "false",
          static_cast<int>(this->metalness.size()),
          this->metalness.empty() ? 0.0f : this->metalness[0],
          static_cast<int>(this->roughness.size()),
          this->roughness.empty() ? 0.5f : this->roughness[0],
          this->roughnessStrength, this->normalStrength, this->emissiveIntensity,
          this->transmissionIor, this->transmissionAbsorption);
}
