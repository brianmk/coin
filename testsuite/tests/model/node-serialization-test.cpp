// testsuite/node-serialization-test.cpp
//
// Host round-trip tests for the two new material/texture nodes:
// write -> read through SoOutput/SoInput must reproduce every authored field.
// New Coin nodes are frequently added without this lock, and a missing field
// registration on read silently drops authored data.  Covers:
//   * SoPhysicalMaterial: metalness, roughness, enabled (MF arrays),
//     roughnessStrength, normalStrength, emissiveIntensity,
//   * SoTextureCoordinateProjection: mapping, scale, offset.
//
// No GPU or window system is involved.

#include <Inventor/SoDB.h>
#include <Inventor/SoInput.h>
#include <Inventor/SoOutput.h>
#include <Inventor/actions/SoSearchAction.h>
#include <Inventor/actions/SoWriteAction.h>
#include <Inventor/nodes/SoPhysicalMaterial.h>
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/nodes/SoTextureCoordinateProjection.h>

#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>

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
  if (std::fabs(actual - expected) > 1.0e-5f) {
    std::cerr << "FAIL: " << message << " (actual=" << actual
              << ", expected=" << expected << ")" << std::endl;
    ++failures;
  }
}

const char * const kPath = "coin-new-nodes-io-test.iv";

// Writes root to a temp file and reads it back.  Returns the read root, owned
// by the caller, or nullptr on failure.
SoSeparator *
roundTrip(SoNode * root)
{
  SoOutput out;
  if (!out.openFile(kPath)) {
    std::cerr << "FAIL: could not open " << kPath << " for writing" << std::endl;
    ++failures;
    return nullptr;
  }
  SoWriteAction write(&out);
  write.apply(root);
  out.closeFile();

  SoInput in;
  if (!in.openFile(kPath)) {
    std::cerr << "FAIL: could not open " << kPath << " for reading" << std::endl;
    ++failures;
    return nullptr;
  }
  SoSeparator * read = SoDB::readAll(&in);
  in.closeFile();
  std::remove(kPath);
  if (!read) {
    std::cerr << "FAIL: SoDB::readAll returned null" << std::endl;
    ++failures;
    return nullptr;
  }
  // SoDB::readAll returns a caller-owned node with reference count 0; take a
  // reference so actions and the unref() at the end are well defined.
  read->ref();
  return read;
}

template <typename NodeT>
NodeT *
findNode(SoNode * root)
{
  SoSearchAction search;
  search.setType(NodeT::getClassTypeId());
  search.setInterest(SoSearchAction::FIRST);
  search.apply(root);
  SoPath * path = search.getPath();
  if (!path) {
    return nullptr;
  }
  return static_cast<NodeT *>(path->getTail());
}

}  // namespace

int
main()
{
  SoDB::init();

  // --- SoPhysicalMaterial -------------------------------------------------
  {
    SoSeparator * root = new SoSeparator;
    root->ref();

    SoPhysicalMaterial * material = new SoPhysicalMaterial;
    material->metalness.setNum(3);
    material->metalness.set1Value(0, 0.1f);
    material->metalness.set1Value(1, 0.5f);
    material->metalness.set1Value(2, 0.9f);
    material->roughness.setNum(3);
    material->roughness.set1Value(0, 0.2f);
    material->roughness.set1Value(1, 0.4f);
    material->roughness.set1Value(2, 0.6f);
    material->enabled.setNum(3);
    material->enabled.set1Value(0, TRUE);
    material->enabled.set1Value(1, FALSE);
    material->enabled.set1Value(2, TRUE);
    material->roughnessStrength.setValue(0.75f);
    material->normalStrength.setValue(0.5f);
    material->emissiveIntensity.setValue(2.25f);
    root->addChild(material);

    SoSeparator * read = roundTrip(root);
    root->unref();

    if (read) {
      SoPhysicalMaterial * got = findNode<SoPhysicalMaterial>(read);
      check(got != nullptr, "SoPhysicalMaterial did not survive the round-trip");
      if (got) {
        check(got->metalness.getNum() == 3 &&
                got->roughness.getNum() == 3 &&
                got->enabled.getNum() == 3,
              "SoPhysicalMaterial MF array lengths not restored");
        checkFloat(got->metalness[0], 0.1f, "metalness[0]");
        checkFloat(got->metalness[1], 0.5f, "metalness[1]");
        checkFloat(got->metalness[2], 0.9f, "metalness[2]");
        checkFloat(got->roughness[0], 0.2f, "roughness[0]");
        checkFloat(got->roughness[1], 0.4f, "roughness[1]");
        checkFloat(got->roughness[2], 0.6f, "roughness[2]");
        check(got->enabled[0] == TRUE && got->enabled[1] == FALSE &&
                got->enabled[2] == TRUE,
              "enabled array not restored");
        checkFloat(got->roughnessStrength.getValue(), 0.75f,
                   "roughnessStrength");
        checkFloat(got->normalStrength.getValue(), 0.5f, "normalStrength");
        checkFloat(got->emissiveIntensity.getValue(), 2.25f,
                   "emissiveIntensity");
      }
      read->unref();
    }
  }

  // --- SoTextureCoordinateProjection -------------------------------------
  {
    SoSeparator * root = new SoSeparator;
    root->ref();

    SoTextureCoordinateProjection * projection =
      new SoTextureCoordinateProjection;
    projection->mapping = SoTextureCoordinateProjection::CYLINDRICAL;
    projection->scale.setValue(2.0f, 3.0f, 4.0f);
    projection->offset.setValue(0.25f, -0.5f, 1.5f);
    root->addChild(projection);

    SoSeparator * read = roundTrip(root);
    root->unref();

    if (read) {
      SoTextureCoordinateProjection * got =
        findNode<SoTextureCoordinateProjection>(read);
      check(got != nullptr,
            "SoTextureCoordinateProjection did not survive the round-trip");
      if (got) {
        check(got->mapping.getValue() ==
                SoTextureCoordinateProjection::CYLINDRICAL,
              "mapping not restored");
        checkFloat(got->scale.getValue()[0], 2.0f, "scale x");
        checkFloat(got->scale.getValue()[1], 3.0f, "scale y");
        checkFloat(got->scale.getValue()[2], 4.0f, "scale z");
        checkFloat(got->offset.getValue()[0], 0.25f, "offset x");
        checkFloat(got->offset.getValue()[1], -0.5f, "offset y");
        checkFloat(got->offset.getValue()[2], 1.5f, "offset z");
      }
      read->unref();
    }
  }

  SoDB::finish();

  if (failures != 0) {
    std::cerr << failures << " check(s) failed" << std::endl;
  }
  return failures == 0 ? 0 : 1;
}
