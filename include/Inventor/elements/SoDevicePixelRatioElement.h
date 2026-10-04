#ifndef COIN_SODEVICEPIXELRATIOELEMENT_H
#define COIN_SODEVICEPIXELRATIOELEMENT_H

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
