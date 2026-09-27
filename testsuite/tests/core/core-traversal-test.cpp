#include <Inventor/SoDB.h>
#include <Inventor/SbViewportRegion.h>
#include <Inventor/SbVec3f.h>
#include <Inventor/actions/SoCallbackAction.h>
#include <Inventor/actions/SoGetPrimitiveCountAction.h>
#include <Inventor/actions/SoRayPickAction.h>
#include <Inventor/nodes/SoCoordinate3.h>
#include <Inventor/nodes/SoCube.h>
#include <Inventor/nodes/SoLineSet.h>
#include <Inventor/nodes/SoPointSet.h>
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/nodes/SoShape.h>

#include <cassert>

namespace {

struct CallbackCounts {
  int triangles = 0;
  int lines = 0;
  int points = 0;
};

void
triangle_callback(void * userdata, SoCallbackAction *,
                  const SoPrimitiveVertex *, const SoPrimitiveVertex *,
                  const SoPrimitiveVertex *)
{
  static_cast<CallbackCounts *>(userdata)->triangles++;
}

void
line_callback(void * userdata, SoCallbackAction *,
              const SoPrimitiveVertex *, const SoPrimitiveVertex *)
{
  static_cast<CallbackCounts *>(userdata)->lines++;
}

void
point_callback(void * userdata, SoCallbackAction *, const SoPrimitiveVertex *)
{
  static_cast<CallbackCounts *>(userdata)->points++;
}

}

int
main()
{
  SoDB::init();

  SoSeparator * root = new SoSeparator;
  SoCube * cube = new SoCube;
  root->addChild(cube);

  SoCoordinate3 * coordinates = new SoCoordinate3;
  coordinates->point.set1Value(0, SbVec3f(-2.0f, 0.0f, 0.0f));
  coordinates->point.set1Value(1, SbVec3f(-1.0f, 0.0f, 0.0f));
  coordinates->point.set1Value(2, SbVec3f(2.0f, 0.0f, 0.0f));
  root->addChild(coordinates);

  SoLineSet * lines = new SoLineSet;
  lines->numVertices.set1Value(0, 2);
  root->addChild(lines);

  SoPointSet * points = new SoPointSet;
  points->numPoints = 1;
  root->addChild(points);
  root->ref();

  {
    CallbackCounts callbacks;
    SoCallbackAction callback_action;
    callback_action.addTriangleCallback(SoShape::getClassTypeId(),
                                        triangle_callback, &callbacks);
    callback_action.addLineSegmentCallback(SoShape::getClassTypeId(),
                                           line_callback, &callbacks);
    callback_action.addPointCallback(SoShape::getClassTypeId(),
                                     point_callback, &callbacks);
    callback_action.apply(root);
    assert(callbacks.triangles > 0);
    assert(callbacks.lines > 0);
    assert(callbacks.points > 0);

    SoGetPrimitiveCountAction count_action;
    count_action.apply(root);
    assert(count_action.getTriangleCount() > 0);
    assert(count_action.getLineCount() > 0);
    assert(count_action.getPointCount() > 0);

    SoRayPickAction pick_action(SbViewportRegion(100, 100));
    pick_action.setRay(SbVec3f(0.0f, 0.0f, 5.0f),
                       SbVec3f(0.0f, 0.0f, -1.0f));
    pick_action.apply(root);
    assert(pick_action.getPickedPoint() != NULL);
  }

  root->unref();
  SoDB::finish();
  return 0;
}
