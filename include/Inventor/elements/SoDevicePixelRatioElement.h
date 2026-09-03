#ifndef COIN_SODEVICEPIXELRATIOELEMENT_H
#define COIN_SODEVICEPIXELRATIOELEMENT_H

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

#include <Inventor/elements/SoFloatElement.h>

class COIN_DLL_API SoDevicePixelRatioElement : public SoFloatElement {
  typedef SoFloatElement inherited;

  SO_ELEMENT_HEADER(SoDevicePixelRatioElement);

public:
  static void initClass(void);

  void init(SoState * state) override;

  static void set(SoState * state, SoNode * node, float dpr);
  static void set(SoState * state, float dpr);
  static float get(SoState * state);

protected:
  ~SoDevicePixelRatioElement() override;
};

#endif // !COIN_SODEVICEPIXELRATIOELEMENT_H
