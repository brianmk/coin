#include <Inventor/SoDB.h>
#include <Inventor/actions/SoCallbackAction.h>
#include <Inventor/elements/SoLazyElement.h>
#include <Inventor/system/gl.h>

#include <iostream>

int
main()
{
  SoDB::init();

  SoCallbackAction action;
  SoState * state = action.getState();
  SoLazyElement::enableSeparateBlending(state,
                                        GL_ZERO, GL_ZERO,
                                        GL_ZERO, GL_ZERO);

  int src = -1;
  int dst = -1;
  if (!SoLazyElement::getAlphaBlending(state, src, dst) ||
      src != GL_ZERO || dst != GL_ZERO) {
    std::cerr << "FAIL: explicit ZERO/ZERO alpha blending was lost" << std::endl;
    SoDB::finish();
    return 1;
  }

  SoLazyElement::enableBlending(state, GL_ONE, GL_ZERO);
  if (SoLazyElement::getAlphaBlending(state, src, dst)) {
    std::cerr << "FAIL: ordinary blending was reported as separate blending" << std::endl;
    SoDB::finish();
    return 1;
  }

  SoDB::finish();
  return 0;
}
