#ifndef COIN_SOACTIONGLP_H
#define COIN_SOACTIONGLP_H

/* This header is private to Coin's implementation. */
#ifndef COIN_INTERNAL
#error this is a private header file
#endif

/* Core builds retain the action and node APIs, but do not register or call
   legacy fixed-function GL elements.  Keep this switch out of public
   headers so platform GL headers cannot leak into application code. */
#if COIN_BUILD_LEGACY_GL_RENDERER
#define SO_ENABLE_LEGACY_GL(action, element) SO_ENABLE(action, element)
#else
#define SO_ENABLE_LEGACY_GL(action, element) do { } WHILE_0
#endif

#endif /* COIN_SOACTIONGLP_H */
