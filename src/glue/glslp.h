#ifndef COIN_GLUE_GLSLP_H
#define COIN_GLUE_GLSLP_H

/**************************************************************************\
 * Copyright (c) Kongsberg Oil & Gas Technologies AS
 * All rights reserved.
 *
 * This file is part of Coin, a 3D graphics library.
\**************************************************************************/

#ifndef COIN_INTERNAL
#error this is a private header file
#endif

#include <Inventor/C/glue/gl.h>

#ifdef __cplusplus
extern "C" {
#endif

GLuint cc_glglue_glCreateShader(const cc_glglue * glue, GLenum type);
void cc_glglue_glShaderSource(const cc_glglue * glue, GLuint shader,
                              GLsizei count, const char * const * string,
                              const GLint * length);
void cc_glglue_glCompileShader(const cc_glglue * glue, GLuint shader);
void cc_glglue_glGetShaderiv(const cc_glglue * glue, GLuint shader,
                             GLenum pname, GLint * params);
void cc_glglue_glGetShaderInfoLog(const cc_glglue * glue, GLuint shader,
                                  GLsizei maxLength, GLsizei * length,
                                  char * infoLog);
void cc_glglue_glDeleteShader(const cc_glglue * glue, GLuint shader);
void cc_glglue_glAttachShader(const cc_glglue * glue, GLuint program,
                              GLuint shader);
void cc_glglue_glDetachShader(const cc_glglue * glue, GLuint program,
                              GLuint shader);
GLint cc_glglue_glGetUniformLocation(const cc_glglue * glue, GLuint program,
                                     const char * name);
void cc_glglue_glGetActiveUniform(const cc_glglue * glue, GLuint program,
                                  GLuint index, GLsizei maxLength,
                                  GLsizei * length, GLint * size,
                                  GLenum * type, char * name);
GLuint cc_glglue_glCreateProgram(const cc_glglue * glue);
void cc_glglue_glLinkProgram(const cc_glglue * glue, GLuint program);
void cc_glglue_glUseProgram(const cc_glglue * glue, GLuint program);
void cc_glglue_glDeleteProgram(const cc_glglue * glue, GLuint program);
void cc_glglue_glGetGLSLProgramiv(const cc_glglue * glue, GLuint program,
                                  GLenum pname, GLint * params);
void cc_glglue_glGetProgramInfoLog(const cc_glglue * glue, GLuint program,
                                   GLsizei maxLength, GLsizei * length,
                                   char * infoLog);
void cc_glglue_glProgramParameteriEXT(const cc_glglue * glue, GLuint program,
                                      GLenum pname, GLint value);

#ifdef __cplusplus
}
#endif

#endif /* !COIN_GLUE_GLSLP_H */
