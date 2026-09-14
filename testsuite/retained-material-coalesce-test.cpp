// testsuite/retained-material-coalesce-test.cpp
//
// Pure-IR test for SoShape's adjacent equal-material batch coalescing.
//
// FreeCAD's SoBrepFaceSet/SoBrepEdgeSet assign a distinct material index per
// face/edge even when the per-face colour array repeats, so a naive
// one-command-per-index emission turns a per-face colour array into one draw
// per face -- thousands of tiny draws on a heavy model.  SoShape coalesces
// adjacent batches whose *resolved* material is identical, which must:
//   - collapse a run of identical materials to a single draw,
//   - leave genuinely distinct materials untouched,
//   - merge only *adjacent* runs (red,red,green,green,green,blue -> 3 draws).
//
// SoCube honours SoMaterialBinding::PER_FACE by assigning material index i to
// face i (via SoCube::generatePrimitives -> SOGEN_MATERIAL_PER_PART), so a cube
// with six diffuse values is a six-batch producer.  No GPU/backend is needed.

#include <Inventor/SoDB.h>
#include <Inventor/SbViewportRegion.h>
#include <Inventor/actions/SoIRRenderAction.h>
#include <Inventor/nodes/SoCube.h>
#include <Inventor/nodes/SoMaterial.h>
#include <Inventor/nodes/SoMaterialBinding.h>
#include <Inventor/nodes/SoSeparator.h>

#include <iostream>
#include <vector>

namespace {

// Apply the IR render action to a PER_FACE cube carrying one diffuse colour
// per face and return the number of draw commands it emitted.  Six faces with
// distinct raw material indices reach the emitter as six batches; the emitter
// is what coalesces them.
int
countCommands(const std::vector<SbColor> & diffuse)
{
  SoSeparator * root = new SoSeparator;
  root->ref();

  SoMaterialBinding * binding = new SoMaterialBinding;
  binding->value = SoMaterialBinding::PER_FACE;
  root->addChild(binding);

  SoMaterial * material = new SoMaterial;
  material->diffuseColor.setNum(static_cast<int>(diffuse.size()));
  for (int i = 0; i < static_cast<int>(diffuse.size()); ++i) {
    material->diffuseColor.set1Value(i, diffuse[i]);
  }
  root->addChild(material);

  root->addChild(new SoCube);

  SoIRRenderAction action(SbViewportRegion(64, 64));
  action.apply(root);

  const int count = action.getDrawList().getNumCommands();
  root->unref();
  return count;
}

} // namespace

int
main()
{
  SoDB::init();

  int failures = 0;

  // Six identical materials: every face resolves to the same material, so the
  // six per-face batches must collapse to a single draw.
  {
    const std::vector<SbColor> same(6, SbColor(1.0f, 0.0f, 0.0f));
    const int n = countCommands(same);
    if (n != 1) {
      std::cerr << "FAIL: six identical materials did not coalesce (commands="
                << n << ", expected 1)" << std::endl;
      ++failures;
    }
  }

  // Six distinct materials: nothing may be merged.
  {
    std::vector<SbColor> distinct;
    for (int i = 0; i < 6; ++i) {
      distinct.push_back(SbColor(static_cast<float>(i) / 6.0f, 0.0f, 0.0f));
    }
    const int n = countCommands(distinct);
    if (n != 6) {
      std::cerr << "FAIL: distinct materials were coalesced (commands=" << n
                << ", expected 6)" << std::endl;
      ++failures;
    }
  }

  // Adjacent runs only: the six faces resolve to
  // red,red,green,green,green,blue -> three draws, in that order.
  {
    std::vector<SbColor> runs;
    runs.push_back(SbColor(1.0f, 0.0f, 0.0f));
    runs.push_back(SbColor(1.0f, 0.0f, 0.0f));
    runs.push_back(SbColor(0.0f, 1.0f, 0.0f));
    runs.push_back(SbColor(0.0f, 1.0f, 0.0f));
    runs.push_back(SbColor(0.0f, 1.0f, 0.0f));
    runs.push_back(SbColor(0.0f, 0.0f, 1.0f));
    const int n = countCommands(runs);
    if (n != 3) {
      std::cerr << "FAIL: adjacent equal-material runs did not coalesce "
                << "(commands=" << n << ", expected 3)" << std::endl;
      ++failures;
    }
  }

  SoDB::finish();
  return failures == 0 ? 0 : 1;
}
