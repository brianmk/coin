#ifndef COIN_SOTRANSPARENCYTYPE_H
#define COIN_SOTRANSPARENCYTYPE_H

/**************************************************************************\
 * Copyright (c) Kongsberg Oil & Gas Technologies AS
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 * 
 * Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 
 * Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 * 
 * Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
\**************************************************************************/

#include <Inventor/fields/SoSFEnum.h>
#include <Inventor/nodes/SoSubNode.h>
class COIN_DLL_API SoTransparencyType : public SoNode {
  typedef SoNode inherited;

  SO_NODE_HEADER(SoTransparencyType);

public:
  static void initClass(void);
  SoTransparencyType(void);

  enum Type {
    SCREEN_DOOR = 0,
    ADD = 1,
    DELAYED_ADD = 2,
    SORTED_OBJECT_ADD = 3,
    BLEND = 4,
    DELAYED_BLEND = 5,
    SORTED_OBJECT_BLEND = 6,
    SORTED_OBJECT_SORTED_TRIANGLE_ADD = 7,
    SORTED_OBJECT_SORTED_TRIANGLE_BLEND = 8,
    NONE = 9
  };

  SoSFEnum value;

  void doAction(SoAction * action) override;
#if COIN_HAVE_LEGACY_GL_RENDERER
  void GLRender(SoGLRenderAction * action) override;
#endif
  void callback(SoCallbackAction * action) override;

protected:
  virtual ~SoTransparencyType();
};

#endif // !COIN_SOTRANSPARENCYTYPE_H
