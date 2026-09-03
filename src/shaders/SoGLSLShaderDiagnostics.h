#ifndef COIN_SOGLSLSHADERDIAGNOSTICS_H
#define COIN_SOGLSLSHADERDIAGNOSTICS_H

// Private helpers shared by the GLSL shader and program implementations.

#include <Inventor/SbString.h>

#include "shaders/SoGLShaderObject.h"
#include "glue/glp.h"
#include "glue/glslp.h"

#include <vector>

static inline const char *
soglsl_stage_name(const SoGLShaderObject::ShaderType type)
{
  switch (type) {
  case SoGLShaderObject::VERTEX:
    return "vertex shader";
  case SoGLShaderObject::FRAGMENT:
    return "fragment shader";
  case SoGLShaderObject::GEOMETRY:
    return "geometry shader";
  default:
    return "shader";
  }
}

static inline SbString
soglsl_get_info_log(const cc_glglue * glue,
                    const GLuint handle,
                    const SbBool program)
{
  GLint length = 0;
  if (program) {
    cc_glglue_glGetGLSLProgramiv(glue, handle, GL_INFO_LOG_LENGTH, &length);
  }
  else {
    cc_glglue_glGetShaderiv(glue, handle, GL_INFO_LOG_LENGTH, &length);
  }

  if (length <= 1) return SbString();

  std::vector<COIN_GLchar> infoLog(static_cast<size_t>(length), '\0');
  GLsizei charsWritten = 0;
  if (program) {
    cc_glglue_glGetProgramInfoLog(glue, handle, length, &charsWritten,
                                  infoLog.data());
  }
  else {
    cc_glglue_glGetShaderInfoLog(glue, handle, length, &charsWritten,
                                 infoLog.data());
  }

  if (charsWritten >= 0 && charsWritten < length) {
    infoLog[static_cast<size_t>(charsWritten)] = '\0';
  }
  else {
    infoLog.back() = '\0';
  }
  return SbString(infoLog.data());
}

#endif /* ! COIN_SOGLSLSHADERDIAGNOSTICS_H */
