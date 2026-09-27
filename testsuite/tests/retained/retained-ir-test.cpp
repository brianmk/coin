#include <Inventor/SoDB.h>
#include <Inventor/SbViewportRegion.h>
#include <Inventor/actions/SoIRRenderAction.h>
#include <Inventor/nodes/SoSeparator.h>

#include <iostream>

static int
runTest()
{
  SoIRRenderAction action(SbViewportRegion(64, 64));
  SoSeparator * root = new SoSeparator;
  root->ref();

  int result = 0;
  action.apply(root);
  if (action.getDrawList().getNumCommands() != 0) {
    std::cerr << "FAIL: empty scene emitted retained commands" << std::endl;
    result = 1;
  }

  SoRenderCommand command;
  command.geometry.topology = SO_TOPOLOGY_TRIANGLES;
  command.geometry.vertexCount = 3;
  command.geometry.vertexStride = sizeof(float) * 3;
  command.material.diffuse.setValue(1.0f, 0.0f, 0.0f, 1.0f);
  action.getMutableDrawList().addCommand(command);
  if (action.getDrawList().getNumCommands() != 1 ||
      action.getDrawList().getCommand(0).geometry.vertexCount != 3) {
    std::cerr << "FAIL: retained command was not stored" << std::endl;
    result = 1;
  }

  root->unref();
  return result;
}

int
main()
{
  SoDB::init();
  const int result = runTest();
  SoDB::finish();
  return result;
}
