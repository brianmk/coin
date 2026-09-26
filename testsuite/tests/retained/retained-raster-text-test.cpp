#include <Inventor/SoDB.h>
#include <Inventor/SbViewportRegion.h>
#include <Inventor/actions/SoIRRenderAction.h>
#include <Inventor/nodes/SoCube.h>
#include <Inventor/nodes/SoDrawStyle.h>
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/nodes/SoText2.h>

#include <iostream>

int
main()
{
  SoDB::init();

  int result = 0;
  SoRenderCommand point;
  point.geometry.topology = SO_TOPOLOGY_POINTS;
  point.state.raster.pointShape = SO_POINT_SHAPE_ROUND;
  point.state.raster.pointSize = 6.0f;
  if (point.state.raster.pointShape != SO_POINT_SHAPE_ROUND ||
      point.state.raster.pointSize != 6.0f) {
    std::cerr << "FAIL: retained point raster state lost its semantic shape or size" << std::endl;
    result = 1;
  }

  SoSeparator * root = new SoSeparator;
  root->ref();
  SoText2 * text = new SoText2;
  text->string = "Coin";
  root->addChild(text);

  SoDrawStyle * drawStyle = new SoDrawStyle;
  drawStyle->style = SoDrawStyle::LINES;
  drawStyle->linePattern = 0x0f0f;
  drawStyle->linePatternScaleFactor = 3;
  root->addChild(drawStyle);
  root->addChild(new SoCube);

  SoIRRenderAction action(SbViewportRegion(128, 64));
  action.apply(root);

  bool foundPixelText = false;
  for (int i = 0; i < action.getDrawList().getNumCommands(); ++i) {
    const SoRenderCommand & command = action.getDrawList().getCommand(i);
    if ((command.material.flags & SO_MAT_IS_PIXEL_TEXT) == 0) continue;
    foundPixelText = command.geometry.vertexCount == 6 &&
      command.material.texture.pixels != NULL &&
      command.material.texture.width > 0 && command.material.texture.height > 0;
    break;
  }
  if (!foundPixelText) {
    std::cerr << "FAIL: SoText2 did not emit a complete retained pixel-text command" << std::endl;
    result = 1;
  }

  bool foundLinePattern = false;
  for (int i = 0; i < action.getDrawList().getNumCommands(); ++i) {
    const SoRenderCommand & command = action.getDrawList().getCommand(i);
    if (command.state.raster.fillMode == 1 &&
        command.state.raster.linePattern == 0x0f0f &&
        command.state.raster.linePatternScale == 3) {
      foundLinePattern = true;
      break;
    }
  }
  if (!foundLinePattern) {
    std::cerr << "FAIL: retained line-pattern state was not captured" << std::endl;
    result = 1;
  }

  root->unref();
  SoDB::finish();
  return result;
}
