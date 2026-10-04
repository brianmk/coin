// src/rendering/SoVulkanDebug.h

#ifndef COIN_SOVULKANDEBUG_H
#define COIN_SOVULKANDEBUG_H

#include <cstdarg>
#include <cstdio>

#include <Inventor/errors/SoDebugError.h>

namespace SoVulkanDebug {

//! Route a renderer diagnostic trace through Coin's message machinery instead
//! of a raw stderr write, so the embedding application's error handler sees it.
//! A trailing newline is dropped (the handler terminates the line itself).
inline void
post(const char * fmt, ...)
{
  char buffer[2048];
  va_list args;
  va_start(args, fmt);
  const int n = std::vsnprintf(buffer, sizeof(buffer), fmt, args);
  va_end(args);
  if (n < 0) {
    return;
  }
  int len = (n < static_cast<int>(sizeof(buffer)))
              ? n : static_cast<int>(sizeof(buffer)) - 1;
  while (len > 0 && (buffer[len - 1] == '\n' || buffer[len - 1] == '\r')) {
    --len;
  }
  buffer[len] = '\0';
  SoDebugError::postInfo("SoVulkan", "%s", buffer);
}

} // namespace SoVulkanDebug

#endif // COIN_SOVULKANDEBUG_H
