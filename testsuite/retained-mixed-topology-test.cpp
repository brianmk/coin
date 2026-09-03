#include <Inventor/SoDB.h>
#include <Inventor/SbBox3f.h>
#include <Inventor/SbViewportRegion.h>
#include <Inventor/SoPrimitiveVertex.h>
#include <Inventor/actions/SoAction.h>
#include <Inventor/actions/SoIRRenderAction.h>
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/nodes/SoShape.h>
#include <Inventor/nodes/SoSubNode.h>

#include <iostream>

class MixedPrimitiveShape : public SoShape {
  SO_NODE_HEADER(MixedPrimitiveShape);

public:
  static void initClass(void);
  MixedPrimitiveShape(void);

protected:
  ~MixedPrimitiveShape() override {}
  void generatePrimitives(SoAction * action) override;
  void computeBBox(SoAction * action, SbBox3f & box,
                   SbVec3f & center) override;
};

SO_NODE_SOURCE(MixedPrimitiveShape);

void
MixedPrimitiveShape::initClass(void)
{
  SO_NODE_INIT_CLASS(MixedPrimitiveShape, SoShape, "SoShape");
}

MixedPrimitiveShape::MixedPrimitiveShape(void)
{
  SO_NODE_CONSTRUCTOR(MixedPrimitiveShape);
}

void
MixedPrimitiveShape::generatePrimitives(SoAction * action)
{
  SoPrimitiveVertex a;
  SoPrimitiveVertex b;
  SoPrimitiveVertex c;
  a.setPoint(-1.0f, -1.0f, 0.0f);
  b.setPoint(1.0f, -1.0f, 0.0f);
  c.setPoint(0.0f, 1.0f, 0.0f);
  a.setNormal(0.0f, 0.0f, 1.0f);
  b.setNormal(0.0f, 0.0f, 1.0f);
  c.setNormal(0.0f, 0.0f, 1.0f);

  this->beginShape(action, TRIANGLES);
  this->shapeVertex(&a);
  this->shapeVertex(&b);
  this->shapeVertex(&c);
  this->endShape();

  a.setPoint(-1.0f, 0.0f, 0.0f);
  b.setPoint(1.0f, 0.0f, 0.0f);
  this->beginShape(action, LINES);
  this->shapeVertex(&a);
  this->shapeVertex(&b);
  this->endShape();

  c.setPoint(0.0f, 0.0f, 0.0f);
  this->beginShape(action, POINTS);
  this->shapeVertex(&c);
  this->endShape();
}

void
MixedPrimitiveShape::computeBBox(SoAction *, SbBox3f & box, SbVec3f & center)
{
  box.setBounds(-1.0f, -1.0f, 0.0f, 1.0f, 1.0f, 0.0f);
  center.setValue(0.0f, 0.0f, 0.0f);
}

static int
runTest()
{
  MixedPrimitiveShape::initClass();

  SoSeparator * root = new SoSeparator;
  root->ref();
  MixedPrimitiveShape * shape = new MixedPrimitiveShape;
  root->addChild(shape);

  SoIRRenderAction action(SbViewportRegion(64, 64));
  action.apply(root);

  int result = 0;
  const SoDrawList & drawlist = action.getDrawList();
  if (drawlist.getNumCommands() != 3) {
    std::cerr << "FAIL: mixed-topology shape did not emit three commands"
              << std::endl;
    result = 1;
  }
  else {
    const SoPrimitiveTopology expected[] = {
      SO_TOPOLOGY_TRIANGLES, SO_TOPOLOGY_LINES, SO_TOPOLOGY_POINTS
    };
    const uint32_t counts[] = { 3, 2, 1 };
    for (int i = 0; i < 3; ++i) {
      const SoRenderCommand & command = drawlist.getCommand(i);
      if (command.geometry.topology != expected[i] ||
          command.geometry.vertexCount != counts[i] ||
          command.userData != shape) {
        std::cerr << "FAIL: mixed-topology command " << i
                  << " is incomplete" << std::endl;
        result = 1;
      }
    }
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
