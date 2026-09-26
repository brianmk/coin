// testsuite/physical-material-element-test.cpp
//
// Pure-CPU contract tests for SoPhysicalMaterialElement, the traversal element
// SoPhysicalMaterial installs so a render backend can pick up an authored
// metallic-roughness material alongside the legacy Blinn-Phong appearance.
//
// These lock the semantics every backend relies on:
//   * an absent producer leaves the element inactive (legacy path),
//   * a single-element array applies to every per-face material index,
//   * a multi-element array resolves per index with safe clamping, so
//     physicalValueAt()/physicalEnabledAt() never index out of bounds,
//   * enabled is per face, so one shape can mix legacy and PBR faces,
//   * an empty enabled array means "inactive" even when metalness is set,
//   * separator scoping saves/restores the element like every other element,
//   * matches() coalesces only byte-for-byte identical parameters.
//
// The element is enabled for SoIRRenderAction (the retained IR traversal the
// Vulkan backends share), so readings are taken from SoCallback probe nodes
// placed in the scene under that action.  No GPU or window system is needed.

#include <Inventor/SoDB.h>
#include <Inventor/SbViewportRegion.h>
#include <Inventor/actions/SoAction.h>
#include <Inventor/actions/SoIRRenderAction.h>
#include <Inventor/elements/SoElement.h>
#include <Inventor/elements/SoPhysicalMaterialElement.h>
#include <Inventor/misc/SoState.h>
#include <Inventor/nodes/SoCube.h>
#include <Inventor/nodes/SoCallback.h>
#include <Inventor/nodes/SoPhysicalMaterial.h>
#include <Inventor/nodes/SoSeparator.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void
check(bool condition, const std::string & message)
{
  if (!condition) {
    std::cerr << "FAIL: " << message << std::endl;
    ++failures;
  }
}

void
checkFloat(float actual, float expected, const std::string & message)
{
  if (std::fabs(actual - expected) > 1.0e-6f) {
    std::ostringstream os;
    os << message << " (actual=" << actual << ", expected=" << expected << ")";
    check(false, os.str());
  }
}

// Per-probe snapshot of the element, captured while its stack is live.  A
// copyMatchInfo() clone is kept alive so the matches() contract can be
// exercised after the traversal has unwound.
struct Reading {
  SbBool enabled0 = FALSE;
  float metalness0 = 0.0f;
  float roughness0 = 0.0f;

  SbBool enabled5 = FALSE;
  float metalness5 = 0.0f;
  float roughness5 = 0.0f;

  SbBool enabled999 = FALSE;
  float metalness999 = 0.0f;
  float roughness999 = 0.0f;

  SbBool enabledNeg = FALSE;
  float metalnessNeg = 0.0f;
  float roughnessNeg = 0.0f;

  float roughnessStrength = 0.0f;
  float normalStrength = 0.0f;
  float emissiveIntensity = 0.0f;

  std::shared_ptr<SoElement> matchCopy;
};

void
collect(void * userdata, SoAction * action)
{
  std::vector<Reading> * readings = static_cast<std::vector<Reading> *>(userdata);
  SoState * state = action->getState();

  Reading reading;
  SoPhysicalMaterialElement::get(state, reading.enabled0, reading.metalness0,
                                 reading.roughness0);
  SoPhysicalMaterialElement::get(state, reading.enabled5, reading.metalness5,
                                 reading.roughness5, 5);
  SoPhysicalMaterialElement::get(state, reading.enabled999, reading.metalness999,
                                 reading.roughness999, 999);
  SoPhysicalMaterialElement::get(state, reading.enabledNeg, reading.metalnessNeg,
                                 reading.roughnessNeg, -1);
  SoPhysicalMaterialElement::getStrengths(state, reading.roughnessStrength,
                                          reading.normalStrength,
                                          reading.emissiveIntensity);

  const SoElement * element =
    state->getConstElement(SoPhysicalMaterialElement::getClassStackIndex());
  reading.matchCopy = std::shared_ptr<SoElement>(element->copyMatchInfo());

  readings->push_back(reading);
}

// Appends an SoCallback probe node that snapshots the element when the IR
// traversal reaches this point.
SoCallback *
addProbe(SoSeparator * parent, std::vector<Reading> * readings)
{
  SoCallback * probe = new SoCallback;
  probe->setCallback(collect, readings);
  parent->addChild(probe);
  return probe;
}

// Applies the IR render action; the probes embedded in the scene do the work.
void
traverse(SoNode * root)
{
  SoIRRenderAction action(SbViewportRegion(64, 64));
  action.apply(root);
}

SoPhysicalMaterial *
makePhysical(float metalness, float roughness)
{
  SoPhysicalMaterial * physical = new SoPhysicalMaterial;
  physical->metalness.setValue(metalness);
  physical->roughness.setValue(roughness);
  return physical;
}

}  // namespace

int
main()
{
  SoDB::init();

  // --- Absent producer: inactive with the documented defaults -------------
  {
    SoSeparator * root = new SoSeparator;
    std::vector<Reading> r;
    root->ref();
    root->addChild(new SoCube);
    addProbe(root, &r);
    traverse(root);
    root->unref();

    check(r.size() == 1, "absent node: expected exactly one probe reading");
    if (r.size() == 1) {
      check(!r[0].enabled0, "absent node: enabled must default to false");
      checkFloat(r[0].metalness0, 0.0f, "absent node: metalness default");
      checkFloat(r[0].roughness0, 0.5f, "absent node: roughness default");
      checkFloat(r[0].roughnessStrength, 1.0f, "absent node: roughnessStrength");
      checkFloat(r[0].normalStrength, 1.0f, "absent node: normalStrength");
      checkFloat(r[0].emissiveIntensity, 1.0f, "absent node: emissiveIntensity");
    }
  }

  // --- Single-element arrays apply to every material index ----------------
  {
    SoSeparator * root = new SoSeparator;
    std::vector<Reading> r;
    root->ref();
    SoPhysicalMaterial * physical = makePhysical(0.7f, 0.2f);
    physical->enabled.setValue(TRUE);
    root->addChild(physical);
    root->addChild(new SoCube);
    addProbe(root, &r);
    traverse(root);
    root->unref();

    check(r.size() == 1, "single-element: expected one probe reading");
    if (r.size() == 1) {
      check(r[0].enabled0 && r[0].enabled5 && r[0].enabled999 &&
              r[0].enabledNeg,
            "single-element enabled must apply to every index");
      checkFloat(r[0].metalness5, 0.7f, "single-element metalness at index 5");
      checkFloat(r[0].metalness999, 0.7f,
                 "single-element metalness at index 999");
      checkFloat(r[0].roughness5, 0.2f, "single-element roughness at index 5");
      checkFloat(r[0].roughnessNeg, 0.2f,
                 "single-element roughness at index -1");
    }
  }

  // --- Multi-element arrays resolve per index and clamp safely ------------
  {
    SoSeparator * root = new SoSeparator;
    std::vector<Reading> r;
    root->ref();
    SoPhysicalMaterial * physical = new SoPhysicalMaterial;
    physical->metalness.setNum(3);
    physical->metalness.set1Value(0, 0.1f);
    physical->metalness.set1Value(1, 0.2f);
    physical->metalness.set1Value(2, 0.3f);
    physical->roughness.setNum(3);
    physical->roughness.set1Value(0, 0.9f);
    physical->roughness.set1Value(1, 0.8f);
    physical->roughness.set1Value(2, 0.7f);
    physical->enabled.setNum(3);
    physical->enabled.set1Value(0, FALSE);
    physical->enabled.set1Value(1, TRUE);
    physical->enabled.set1Value(2, TRUE);
    root->addChild(physical);
    root->addChild(new SoCube);
    addProbe(root, &r);
    traverse(root);
    root->unref();

    check(r.size() == 1, "multi-element: expected one probe reading");
    if (r.size() == 1) {
      checkFloat(r[0].metalness0, 0.1f, "multi-element index 0");
      checkFloat(r[0].roughness0, 0.9f, "multi-element roughness index 0");
      check(!r[0].enabled0, "multi-element enabled index 0 is false");
      // Index 5 is past the end and must clamp to the last entry (index 2).
      checkFloat(r[0].metalness5, 0.3f, "multi-element index 5 clamps to last");
      checkFloat(r[0].roughness5, 0.7f,
                 "multi-element roughness index 5 clamps");
      check(r[0].enabled5, "multi-element enabled index 5 clamps to last");
      checkFloat(r[0].metalness999, 0.3f,
                 "multi-element index 999 clamps to last");
      check(r[0].enabled999, "multi-element enabled index 999 clamps");
      // Negative index must clamp to the first entry (index 0).
      checkFloat(r[0].metalnessNeg, 0.1f,
                 "multi-element index -1 clamps to first");
      check(!r[0].enabledNeg, "multi-element enabled index -1 clamps to first");
    }
  }

  // --- enabled FALSE (and empty) forces the legacy path -------------------
  {
    SoSeparator * root = new SoSeparator;
    std::vector<Reading> r;
    root->ref();
    SoPhysicalMaterial * disabled = makePhysical(0.9f, 0.05f);
    disabled->enabled.setValue(FALSE);
    root->addChild(disabled);
    root->addChild(new SoCube);
    addProbe(root, &r);
    traverse(root);
    root->unref();

    check(r.size() == 1, "disabled: expected one probe reading");
    if (r.size() == 1) {
      check(!r[0].enabled0,
            "enabled FALSE must force the legacy path despite metalness");
      checkFloat(r[0].metalness0, 0.9f,
                 "enabled FALSE still carries the authored metalness");
    }
  }
  {
    SoSeparator * root = new SoSeparator;
    std::vector<Reading> r;
    root->ref();
    SoPhysicalMaterial * empty = makePhysical(0.9f, 0.05f);
    empty->enabled.setNum(0);
    root->addChild(empty);
    root->addChild(new SoCube);
    addProbe(root, &r);
    traverse(root);
    root->unref();

    check(r.size() == 1, "empty-enabled: expected one probe reading");
    if (r.size() == 1) {
      check(!r[0].enabled0,
            "an empty enabled array must force the legacy path");
    }
  }

  // --- Separator scoping: save/restore and sibling isolation --------------
  {
    // root: physicalA, sep{ physicalB, cube1, probe }, cube2, probe2
    // The first probe sees physicalB, the second must see the restored
    // physicalA.
    SoSeparator * root = new SoSeparator;
    std::vector<Reading> r;
    root->ref();
    root->addChild(makePhysical(0.1f, 0.6f));

    SoSeparator * branch = new SoSeparator;
    branch->addChild(makePhysical(0.2f, 0.7f));
    branch->addChild(new SoCube);
    addProbe(branch, &r);
    root->addChild(branch);

    root->addChild(new SoCube);
    addProbe(root, &r);
    traverse(root);
    root->unref();

    check(r.size() == 2, "scoping: expected two probe readings");
    if (r.size() == 2) {
      checkFloat(r[0].metalness0, 0.2f, "nested branch sees its own material");
      checkFloat(r[0].roughness0, 0.7f, "nested branch sees its own roughness");
      checkFloat(r[1].metalness0, 0.1f,
                 "leaving a branch restores the parent material");
      checkFloat(r[1].roughness0, 0.6f,
                 "leaving a branch restores the parent roughness");
    }
  }
  {
    // A plain sibling branch must not inherit a material set in another
    // sibling branch.
    SoSeparator * root = new SoSeparator;
    std::vector<Reading> r;
    root->ref();

    SoSeparator * physicalBranch = new SoSeparator;
    physicalBranch->addChild(makePhysical(0.8f, 0.1f));
    physicalBranch->addChild(new SoCube);
    addProbe(physicalBranch, &r);
    root->addChild(physicalBranch);

    SoSeparator * plainBranch = new SoSeparator;
    plainBranch->addChild(new SoCube);
    addProbe(plainBranch, &r);
    root->addChild(plainBranch);

    traverse(root);
    root->unref();

    check(r.size() == 2, "sibling isolation: expected two probe readings");
    if (r.size() == 2) {
      check(r[0].enabled0, "physical branch sees the PBR material");
      check(!r[1].enabled0,
            "plain sibling branch must not inherit the PBR material");
    }
  }

  // --- matches(): identical parameters coalesce, a changed one does not ---
  {
    const auto twoReadings = [](float strengthA, float strengthB) {
      SoSeparator * root = new SoSeparator;
      std::vector<Reading> r;
      root->ref();
      SoSeparator * first = new SoSeparator;
      SoPhysicalMaterial * a = makePhysical(0.4f, 0.35f);
      a->roughnessStrength.setValue(strengthA);
      first->addChild(a);
      first->addChild(new SoCube);
      addProbe(first, &r);
      root->addChild(first);

      SoSeparator * second = new SoSeparator;
      SoPhysicalMaterial * b = makePhysical(0.4f, 0.35f);
      b->roughnessStrength.setValue(strengthB);
      second->addChild(b);
      second->addChild(new SoCube);
      addProbe(second, &r);
      root->addChild(second);

      traverse(root);
      root->unref();
      return r;
    };

    const std::vector<Reading> same = twoReadings(1.0f, 1.0f);
    check(same.size() == 2, "matches: expected two probe readings");
    if (same.size() == 2) {
      check(same[0].matchCopy->matches(same[1].matchCopy.get()),
            "identical physical parameters must coalesce");
    }

    const std::vector<Reading> changed = twoReadings(1.0f, 0.25f);
    check(changed.size() == 2, "matches: expected two probe readings");
    if (changed.size() == 2) {
      check(!changed[0].matchCopy->matches(changed[1].matchCopy.get()),
            "a changed roughnessStrength must not coalesce");
    }
  }

  SoDB::finish();

  if (failures != 0) {
    std::cerr << failures << " check(s) failed" << std::endl;
  }
  return failures == 0 ? 0 : 1;
}
