// src/rendering/SoVulkanResult.h
//
// Lightweight status type for Coin's Vulkan renderer internals.  Internal to
// Coin (not installed, not public API).
//
// Motivation: the renderer historically returned bare SbBool from every
// fallible helper and reported the reason by logging as a side effect, so a
// caller could not distinguish "not applicable" from "failed", could not
// propagate a reason, and had to string-match logs to react.  SoVulkanResult
// carries a status and a human-readable reason, so a failure can be
// propagated to a decision point (e.g. drop to raster) without a log parse.
//
// The public SoRenderBackend interface keeps returning SbBool for ABI
// stability; internal entry points may return SoVulkanResult and the
// embedding converts at the boundary.

#ifndef COIN_SOVULKANRESULT_H
#define COIN_SOVULKANRESULT_H

#include <cstdint>
#include <string>
#include <utility>

namespace SoVulkan {

enum class Status : uint8_t {
  Ok = 0,
  Error,            //!< generic failure (allocation, upload, recording)
  InvalidArgument,  //!< caller misuse (null target, bad params)
};

/*!
  \brief A status plus an optional reason.

  Default-constructed is Ok.  Only failures allocate a message; the Ok path is
  a single enum compare.  Copyable and movable.
*/
class Result {
public:
  Result() = default;

  static Result ok()
  {
    return Result {};
  }
  static Result error(std::string message)
  {
    return Result(Status::Error, std::move(message));
  }
  static Result invalidArgument(std::string message)
  {
    return Result(Status::InvalidArgument, std::move(message));
  }

  bool isOk() const { return this->status_ == Status::Ok; }
  explicit operator bool() const { return this->isOk(); }
  Status status() const { return this->status_; }
  const std::string & message() const { return this->message_; }

private:
  Result(Status status, std::string message)
    : status_(status), message_(std::move(message))
  {
  }

  Status status_ = Status::Ok;
  std::string message_;
};

/*!
  \brief Propagate a failed Result out of the current function.

  Usage: `SO_VULKAN_TRY(uploadTexture(...));` where the enclosing function
  returns SoVulkan::Result.  A no-op on success.
*/
#define SO_VULKAN_TRY(expr)                                                    \
  do {                                                                         \
    ::SoVulkan::Result so_vulkan_try_result = (expr);                          \
    if (!so_vulkan_try_result.isOk()) {                                        \
      return so_vulkan_try_result;                                             \
    }                                                                          \
  } while (false)

} // namespace SoVulkan

#endif // COIN_SOVULKANRESULT_H
