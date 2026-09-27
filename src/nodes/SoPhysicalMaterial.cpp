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
  \class SoPhysicalMaterial Inventor/nodes/SoPhysicalMaterial.h
  \brief An authored metallic-roughness material for physically based shading.

  Sets the SoPhysicalMaterialElement for all subsequent shapes, so a render
  backend with a physical shader can pick up metalness/roughness alongside the
  conventional SoMaterial appearance.  Absent (or with enabled FALSE), the
  renderer keeps the legacy Blinn-Phong appearance.

  <b>FILE FORMAT/DEFAULTS:</b>
  \code
    PhysicalMaterial {
        metalness 0
        roughness 0.5
        enabled TRUE
    }
  \endcode

  \sa SoMaterial, SoPhysicalMaterialElement
*/

#include <Inventor/actions/SoCallbackAction.h>
#include <Inventor/elements/SoPhysicalMaterialElement.h>
#include <Inventor/nodes/SoPhysicalMaterial.h>

#include "nodes/SoSubNodeP.h"

#include <vector>

// *************************************************************************

SO_NODE_SOURCE(SoPhysicalMaterial);

/*!
  \var SoSFFloat SoPhysicalMaterial::metalness
  Metalness of the material.  Defaults to 0.
*/
/*!
  \var SoSFFloat SoPhysicalMaterial::roughness
  Perceptual roughness of the material.  Defaults to 0.5.
*/
/*!
  \var SoSFBool SoPhysicalMaterial::enabled
  Whether the physical parameters are active.  Defaults to TRUE.
*/

/*!
  Constructor.
*/
SoPhysicalMaterial::SoPhysicalMaterial(void)
{
  SO_NODE_INTERNAL_CONSTRUCTOR(SoPhysicalMaterial);

  SO_NODE_ADD_FIELD(metalness, (0.0f));
  SO_NODE_ADD_FIELD(roughness, (0.5f));
  SO_NODE_ADD_FIELD(enabled, (TRUE));
  SO_NODE_ADD_FIELD(roughnessStrength, (1.0f));
  SO_NODE_ADD_FIELD(normalStrength, (1.0f));
  SO_NODE_ADD_FIELD(emissiveIntensity, (1.0f));
  SO_NODE_ADD_FIELD(transmissionIor, (1.5f));
  SO_NODE_ADD_FIELD(transmissionAbsorption, (0.0f));
}

/*!
  Destructor.
*/
SoPhysicalMaterial::~SoPhysicalMaterial()
{
}

/*!
  \copybrief SoBase::initClass(void)
*/
void
SoPhysicalMaterial::initClass(void)
{
  SO_NODE_INTERNAL_INIT_CLASS(SoPhysicalMaterial, SO_FROM_INVENTOR_1);
}

//! Sets the physical-material element for the current traversal state.
void
SoPhysicalMaterial::doAction(SoAction * action)
{
  const int numEnabled = this->enabled.getNum();
  const int numMetal = this->metalness.getNum();
  const int numRough = this->roughness.getNum();
  const SbBool * enabledValues = this->enabled.getValues(0);
  const float * metalValues = this->metalness.getValues(0);
  const float * roughValues = this->roughness.getValues(0);

  std::vector<unsigned char> enabled;
  enabled.reserve(static_cast<size_t>(numEnabled));
  for (int i = 0; i < numEnabled; ++i) {
    enabled.push_back(enabledValues[i] ? 1 : 0);
  }

  std::vector<float> metal(metalValues, metalValues + numMetal);
  std::vector<float> rough(roughValues, roughValues + numRough);

  SoPhysicalMaterialElement::set(action->getState(), this,
                                 enabled, metal, rough,
                                 this->roughnessStrength.getValue(),
                                 this->normalStrength.getValue(),
                                 this->emissiveIntensity.getValue(),
                                 this->transmissionIor.getValue(),
                                 this->transmissionAbsorption.getValue());
}

//! Callback traversal hook (mirrors doAction so IR/callback paths agree).
void
SoPhysicalMaterial::callback(SoCallbackAction * action)
{
  SoPhysicalMaterial::doAction(action);
}
