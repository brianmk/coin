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
#include <Inventor/nodes/SoPhysicalMaterial.h>
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

// Same setup, but the six faces share one diffuse colour and differ only in
// their per-face physical (metallic-roughness) parameters.  The resolved
// material comparison must include metalness/roughness, so distinct physical
// materials must not be coalesced even though the colour is identical.
int
countPhysicalCommands(const std::vector<float> & metalness)
{
  SoSeparator * root = new SoSeparator;
  root->ref();

  SoMaterialBinding * binding = new SoMaterialBinding;
  binding->value = SoMaterialBinding::PER_FACE;
  root->addChild(binding);

  SoMaterial * material = new SoMaterial;
  material->diffuseColor.setNum(6);
  for (int i = 0; i < 6; ++i) {
    material->diffuseColor.set1Value(i, SbColor(1.0f, 0.0f, 0.0f));
  }
  root->addChild(material);

  SoPhysicalMaterial * physical = new SoPhysicalMaterial;
  physical->metalness.setNum(static_cast<int>(metalness.size()));
  for (int i = 0; i < static_cast<int>(metalness.size()); ++i) {
    physical->metalness.set1Value(i, metalness[i]);
  }
  physical->enabled.setNum(6);
  for (int i = 0; i < 6; ++i) {
    physical->enabled.set1Value(i, TRUE);
  }
  root->addChild(physical);

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

  // SoPhysicalMaterial must reach the IR material: a physical node enables
  // the PBR flag and carries metalness/roughness; without one the material
  // stays on the legacy Blinn-Phong path.
  {
    SoSeparator * root = new SoSeparator;
    root->ref();
    SoPhysicalMaterial * physical = new SoPhysicalMaterial;
    physical->metalness.setValue(0.85f);
    physical->roughness.setValue(0.2f);
    physical->transmissionIor.setValue(1.62f);
    physical->transmissionAbsorption.setValue(0.35f);
    root->addChild(physical);
    root->addChild(new SoCube);

    SoIRRenderAction action(SbViewportRegion(64, 64));
    action.apply(root);
    const int n = action.getDrawList().getNumCommands();
    if (n < 1) {
      std::cerr << "FAIL: physical-material scene emitted no commands"
                << std::endl;
      ++failures;
    }
    else {
      const SoMaterialData & mat =
        action.getDrawList().getCommand(0).material;
      if (!mat.physicalMaterial || mat.metalness != 0.85f ||
          mat.roughness != 0.2f) {
        std::cerr << "FAIL: SoPhysicalMaterial did not reach the IR "
                  << "(enabled=" << mat.physicalMaterial
                  << ", metalness=" << mat.metalness
                  << ", roughness=" << mat.roughness << ")" << std::endl;
        ++failures;
      }
      if (mat.transmissionIor != 1.62f ||
          mat.transmissionAbsorption != 0.35f) {
        std::cerr << "FAIL: SoPhysicalMaterial optics did not reach the IR "
                  << "(ior=" << mat.transmissionIor
                  << ", absorption=" << mat.transmissionAbsorption << ")"
                  << std::endl;
        ++failures;
      }
      if (!mat.transmissionAuthored) {
        std::cerr << "FAIL: an enabled physical material did not flag its "
                  << "optics as authored (viewer glass would override it)"
                  << std::endl;
        ++failures;
      }
    }
    root->unref();
  }

  {
    SoSeparator * root = new SoSeparator;
    root->ref();
    root->addChild(new SoCube);
    SoIRRenderAction action(SbViewportRegion(64, 64));
    action.apply(root);
    const int n = action.getDrawList().getNumCommands();
    if (n < 1 || action.getDrawList().getCommand(0).material.physicalMaterial) {
      std::cerr << "FAIL: absent SoPhysicalMaterial enabled PBR" << std::endl;
      ++failures;
    }
    else if (action.getDrawList().getCommand(0).material.transmissionAuthored) {
      std::cerr << "FAIL: absent SoPhysicalMaterial flagged optics as authored "
                << "(the viewer glass default must apply instead)" << std::endl;
      ++failures;
    }
    root->unref();
  }

  // Physical materials participate in the resolved-material comparison: faces
  // that differ only in metalness must stay distinct even though their colour
  // is identical, and identical physical runs must still coalesce.
  {
    const std::vector<float> same(6, 0.5f);
    const int n = countPhysicalCommands(same);
    if (n != 1) {
      std::cerr << "FAIL: identical physical materials did not coalesce "
                << "(commands=" << n << ", expected 1)" << std::endl;
      ++failures;
    }
  }
  {
    std::vector<float> distinct;
    for (int i = 0; i < 6; ++i) {
      distinct.push_back(static_cast<float>(i) / 6.0f);
    }
    const int n = countPhysicalCommands(distinct);
    if (n != 6) {
      std::cerr << "FAIL: distinct per-face physical materials were coalesced "
                << "(commands=" << n << ", expected 6)" << std::endl;
      ++failures;
    }
  }
  {
    const std::vector<float> runs = {0.0f, 0.0f, 0.5f, 0.5f, 0.5f, 0.0f};
    const int n = countPhysicalCommands(runs);
    if (n != 3) {
      std::cerr << "FAIL: adjacent equal-physical runs did not coalesce "
                << "(commands=" << n << ", expected 3)" << std::endl;
      ++failures;
    }
  }

  SoDB::finish();
  return failures == 0 ? 0 : 1;
}
