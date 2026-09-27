#include <Inventor/C/glue/gl.h>

#include "glue/glp.h"

#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

int skip(const char * reason)
{
  std::cout << "SKIP: " << reason << std::endl;
  return 77;
}

bool check(bool condition, const char * message)
{
  if (!condition) std::cerr << "FAIL: " << message << std::endl;
  return condition;
}

bool check_no_error(const char * message)
{
  const GLenum error = glGetError();
  if (error == GL_NO_ERROR) return true;
  std::cerr << "FAIL: " << message << " (0x" << std::hex << error
            << std::dec << ")" << std::endl;
  return false;
}

void set_environment(const char * name, const char * value)
{
#ifdef _WIN32
  _putenv_s(name, value);
#else
  setenv(name, value, 1);
#endif
}

void print_shader_log(const cc_glglue * glue, GLuint object, SbBool program)
{
  GLint length = 0;
  if (program) {
    glue->glGetProgramiv(object, GL_INFO_LOG_LENGTH, &length);
  }
  else {
    glue->glGetShaderiv(object, GL_INFO_LOG_LENGTH, &length);
  }

  if (length <= 1) return;

  std::vector<char> log(static_cast<size_t>(length), '\0');
  GLsizei written = 0;
  if (program) {
    glue->glGetProgramInfoLog(object, length, &written, log.data());
  }
  else {
    glue->glGetShaderInfoLog(object, length, &written, log.data());
  }
  std::cerr << "GLSL log: " << log.data() << std::endl;
}

bool compile_shader(const cc_glglue * glue, GLenum type, const char * source,
                    GLuint & shader)
{
  shader = glue->glCreateShader(type);
  if (!check(shader != 0, "profile-neutral shader creation failed")) return false;

  const char * sources[] = { source };
  glue->glShaderSource(shader, 1, sources, NULL);
  glue->glCompileShader(shader);

  GLint compiled = GL_FALSE;
  glue->glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
  if (!check(compiled == GL_TRUE, "profile-neutral shader compilation failed")) {
    print_shader_log(glue, shader, FALSE);
    return false;
  }
  return true;
}

} // namespace

int main()
{
  set_environment("COIN_EGL", "1");
  set_environment("EGL_PLATFORM", "surfaceless");
  set_environment("COIN_EGL_CORE_PROFILE", "1");

  void * context = cc_glglue_context_create_offscreen(16, 16);
  if (context == NULL) return skip("core EGL offscreen context is unavailable");

  if (!cc_glglue_context_make_current(context)) {
    cc_glglue_context_destruct(context);
    return skip("core EGL offscreen context could not be made current");
  }

  const cc_glglue * glue = cc_glglue_instance_from_context_ptr(context);
  GLuint vertexShader = 0;
  GLuint fragmentShader = 0;
  GLuint program = 0;
  int result = 1;

  do {
    static const char vertexSource[] =
      "#version 330 core\n"
      "uniform float u_scale;\n"
      "void main() { gl_Position = vec4(u_scale, 0.0, 0.0, 1.0); }\n";
    static const char fragmentSource[] =
      "#version 330 core\n"
      "out vec4 coinColor;\n"
      "void main() { coinColor = vec4(1.0); }\n";

    if (!check(glue != NULL, "GL glue instance was not created")) break;
    if (!compile_shader(glue, GL_VERTEX_SHADER, vertexSource, vertexShader)) break;
    if (!compile_shader(glue, GL_FRAGMENT_SHADER, fragmentSource, fragmentShader)) break;

    program = glue->glCreateProgram();
    if (!check(program != 0, "profile-neutral program creation failed")) break;
    glue->glAttachShader(program, vertexShader);
    glue->glAttachShader(program, fragmentShader);
    glue->glLinkProgram(program);

    GLint linked = GL_FALSE;
    glue->glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (!check(linked == GL_TRUE, "profile-neutral program linking failed")) {
      print_shader_log(glue, program, TRUE);
      break;
    }

    glue->glUseProgram(program);
    if (!check_no_error("profile-neutral glUseProgram failed")) break;

    const GLint location = glue->glGetUniformLocation(program, "u_scale");
    if (!check(location >= 0, "profile-neutral uniform lookup failed")) break;
    glue->glUniform1f(location, 1.0f);
    if (!check_no_error("profile-neutral uniform update failed")) break;

    result = 0;
  } while (false);

  if (program != 0) {
    glue->glUseProgram(0);
    glue->glDeleteProgram(program);
  }
  if (vertexShader != 0) glue->glDeleteShader(vertexShader);
  if (fragmentShader != 0) glue->glDeleteShader(fragmentShader);

  cc_glglue_context_reinstate_previous(context);
  cc_glglue_context_destruct(context);
  return result;
}
