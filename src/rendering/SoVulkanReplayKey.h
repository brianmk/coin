// src/rendering/SoVulkanReplayKey.h

#ifndef COIN_SOVULKANREPLAYKEY_H
#define COIN_SOVULKANREPLAYKEY_H

/*!
  \file SoVulkanReplayKey.h
  \brief Graph-fingerprint helpers for the Vulkan retained-IR replay.

  Backs SoVulkanRenderManagerP's camera-only-frame replay: a hash over
  render-affecting node ids (mixed with pointer identity) that is invariant under
  camera/light/environment chatter, so a navigation frame keeps the retained draw
  list replayable.  Header-inline; the pure graph walk has no manager dependency.
*/

#include <Inventor/nodes/SoCamera.h>
#include <Inventor/nodes/SoEnvironment.h>
#include <Inventor/nodes/SoGroup.h>
#include <Inventor/nodes/SoLight.h>
#include <Inventor/nodes/SoNode.h>
#include <Inventor/nodes/SoRotation.h>
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/nodes/SoTransformSeparator.h>

#include <cstddef>
#include <cstdint>

namespace CoinVulkanReplay {

//! Order-sensitive hash mix used by the fingerprint walk.
inline void mixHash(uint64_t & h, uint64_t v)
{
  h ^= v + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2);
}

/*!
  \brief True for a node whose id the graph fingerprint deliberately ignores.

  Camera-coupled chatter (camera/light/environment/rotation/transform-separator)
  and plain container nodes.  Coin propagates notifications up the parent chain,
  so any change re-bumps every ancestor's node-id, and FreeCAD re-aims the
  headlight rotation every navigation frame.  None of these produce the retained
  fill geometry (lighting is re-derived per frame), so excluding their ids only
  suppresses camera-coupled chatter; real edits use transform/shape/selection
  nodes, which still fold their ids.  Shared with the scene-dirty sensor.
*/
inline bool fingerprintSkipsNodeId(const SoNode * node)
{
  return node->isOfType(SoCamera::getClassTypeId()) ||
    node->isOfType(SoLight::getClassTypeId()) ||
    node->isOfType(SoEnvironment::getClassTypeId()) ||
    node->isOfType(SoRotation::getClassTypeId()) ||
    node->isOfType(SoTransformSeparator::getClassTypeId()) ||
    node->getTypeId() == SoGroup::getClassTypeId() ||
    node->getTypeId() == SoSeparator::getClassTypeId();
}

//! Fold (node pointer, getNodeId()) of every reachable node into \a h, excluding
//! camera-coupled infra, so a camera-only frame yields an unchanged hash.
inline void graphFingerprintWalk(SoNode * node, const SoNode * skip, uint64_t & h)
{
  if (!node || node == skip) return;
  mixHash(h, reinterpret_cast<uintptr_t>(node));
  if (!fingerprintSkipsNodeId(node)) {
    mixHash(h, static_cast<uint64_t>(node->getNodeId()));
  }
  if (node->isOfType(SoGroup::getClassTypeId())) {
    const SoGroup * group = static_cast<const SoGroup *>(node);
    const int num = group->getNumChildren();
    mixHash(h, static_cast<uint64_t>(num));
    for (int i = 0; i < num; ++i) {
      graphFingerprintWalk(group->getChild(i), skip, h);
    }
  }
}

} // namespace CoinVulkanReplay

#endif // COIN_SOVULKANREPLAYKEY_H
