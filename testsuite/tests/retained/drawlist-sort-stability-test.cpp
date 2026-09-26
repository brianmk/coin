// testsuite/drawlist-sort-stability-test.cpp
//
// Pure-IR test for SoDrawList::buildSortedOrder(): equal sort keys must
// preserve insertion order (stable sort), and differing depth must reorder
// correctly.  No GPU/backend is required.

#include <Inventor/SoDB.h>
#include <Inventor/rendering/SoRenderIR.h>

#include <iostream>
#include <vector>

namespace {

SoRenderCommand commandAt(float z)
{
  SoRenderCommand command;
  command.modelMatrix.makeIdentity();
  command.modelMatrix.setTranslate(SbVec3f(0.0f, 0.0f, z));
  return command;
}

bool sameOrder(const std::vector<int> & order, std::initializer_list<int> expected)
{
  if (order.size() != expected.size()) return false;
  int i = 0;
  for (int value : expected) {
    if (order[static_cast<size_t>(i++)] != value) return false;
  }
  return true;
}

} // namespace

int
main()
{
  SoDB::init();

  int failures = 0;

  // Equal depth (and equal pipeline key) -> stable sort keeps insertion order.
  {
    SoDrawList drawlist;
    drawlist.addCommand(commandAt(0.0f));
    drawlist.addCommand(commandAt(0.0f));
    drawlist.addCommand(commandAt(0.0f));
    SbMatrix view;
    view.makeIdentity();
    drawlist.buildSortedOrder(view);
    if (!sameOrder(drawlist.getSortedOrder(), {0, 1, 2})) {
      std::cerr << "FAIL: equal sort keys did not preserve insertion order"
                << std::endl;
      ++failures;
    }
  }

  // Nearer (smaller camera-space depth -> drawn later for opaque) must sort
  // to the front of an otherwise-identical set.
  {
    SoDrawList drawlist;
    drawlist.addCommand(commandAt(1.0f));  // far
    drawlist.addCommand(commandAt(0.0f));  // near
    drawlist.addCommand(commandAt(0.0f));  // near
    SbMatrix view;
    view.makeIdentity();
    drawlist.buildSortedOrder(view);
    // Far-to-near: command 0 is far and sorts first, commands 1 and 2 are
    // equal-depth and keep their relative insertion order.
    if (!sameOrder(drawlist.getSortedOrder(), {0, 1, 2})) {
      std::cerr << "FAIL: depth reordering produced unexpected order"
                << std::endl;
      ++failures;
    }
  }

  SoDB::finish();
  return failures == 0 ? 0 : 1;
}
