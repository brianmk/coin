/**************************************************************************\
 * Copyright (c) 2026 The Coin3D contributors                          *
 *                                                                        *
 * This file is part of Coin.                                            *
 *                                                                        *
 * Coin is free software; you can redistribute it and/or modify it under *
 * the terms of the GNU General Public License as published by the Free  *
 * Software Foundation; either version 2 of the License, or (at your      *
 * option) any later version.                                            *
\**************************************************************************/

#include <Inventor/misc/SoState.h>

#include <Inventor/elements/SoDevicePixelRatioElement.h>

SO_ELEMENT_SOURCE(SoDevicePixelRatioElement);

void
SoDevicePixelRatioElement::initClass(void)
{
  SO_ELEMENT_INIT_CLASS(SoDevicePixelRatioElement, inherited);
}

void
SoDevicePixelRatioElement::init(SoState * state)
{
  inherited::init(state);
  this->data = 1.0f;
}

void
SoDevicePixelRatioElement::set(SoState * state, SoNode * node, float dpr)
{
  inherited::set(classStackIndex, state, node, dpr);
}

void
SoDevicePixelRatioElement::set(SoState * state, float dpr)
{
  SoDevicePixelRatioElement::set(state, NULL, dpr);
}

float
SoDevicePixelRatioElement::get(SoState * state)
{
  return inherited::get(classStackIndex, state);
}

SoDevicePixelRatioElement::~SoDevicePixelRatioElement(void)
{
}
