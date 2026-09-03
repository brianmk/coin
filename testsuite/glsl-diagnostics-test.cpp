#include <Inventor/SoDB.h>
#include <Inventor/errors/SoDebugError.h>

#include "glue/glp.h"
#include "shaders/SoGLSLShaderObject.h"
#include "shaders/SoGLSLShaderProgram.h"
#include "glue/gl_egl.h"

#include <cstdlib>
#include <iostream>
#include <string>
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

void set_environment(const char * name, const char * value)
{
#ifdef _WIN32
  _putenv_s(name, value);
#else
  setenv(name, value, 1);
#endif
}

struct DiagnosticCapture {
  int warnings = 0;
  int infos = 0;
  std::vector<std::string> messages;
};

void capture_diagnostic(const SoError * error, void * data)
{
  DiagnosticCapture * capture = static_cast<DiagnosticCapture *>(data);
  if (!error->isOfType(SoDebugError::getClassTypeId())) return;

  const SoDebugError * debug = static_cast<const SoDebugError *>(error);
  if (debug->getSeverity() == SoDebugError::WARNING) ++capture->warnings;
  if (debug->getSeverity() == SoDebugError::INFO) ++capture->infos;
  capture->messages.push_back(debug->getDebugString().getString());
}

std::string messages_as_text(const DiagnosticCapture & capture)
{
  std::string result;
  for (std::vector<std::string>::const_iterator it = capture.messages.begin();
       it != capture.messages.end(); ++it) {
    if (!result.empty()) result += "\n";
    result += *it;
  }
  return result;
}

bool contains(const DiagnosticCapture & capture, const char * text)
{
  return messages_as_text(capture).find(text) != std::string::npos;
}

} // namespace

int main()
{
  set_environment("COIN_EGL", "1");
  set_environment("EGL_PLATFORM", "surfaceless");
  set_environment("COIN_EGL_CORE_PROFILE", "1");
  SoDB::init();

  void * context = cc_glglue_context_create_offscreen(16, 16);
  if (context == NULL) {
    SoDB::finish();
    return skip("core EGL offscreen context is unavailable");
  }
  if (!cc_glglue_context_make_current(context)) {
    cc_glglue_context_destruct(context);
    SoDB::finish();
    return skip("core EGL offscreen context could not be made current");
  }

  const cc_glglue * glue = cc_glglue_instance_from_context_ptr(context);
  if (glue == NULL || glue->contextid == 0) {
    cc_glglue_context_reinstate_previous(context);
    cc_glglue_context_destruct(context);
    SoDB::finish();
    return skip("GL glue instance is unavailable");
  }

  SoErrorCB * previousCallback = SoDebugError::getHandlerCallback();
  void * previousData = SoDebugError::getHandlerData();
  int result = 1;

  do {
    DiagnosticCapture compileCapture;
    SoDebugError::setHandlerCallback(capture_diagnostic, &compileCapture);

    SoGLSLShaderObject brokenShader(glue->contextid);
    brokenShader.setShaderType(SoGLShaderObject::VERTEX);
    brokenShader.sourceHint = "broken.vert";
    brokenShader.load("#version 330 core\n"
                      "void main() { this is not valid GLSL; }\n");

    if (!check(!brokenShader.isLoaded(),
               "invalid shader unexpectedly compiled")) break;
    if (!check(compileCapture.warnings > 0,
               "shader compilation failure was not reported as a warning")) break;
    if (!check(contains(compileCapture, "vertex shader"),
               "shader diagnostic omitted its stage")) break;
    if (!check(contains(compileCapture, "broken.vert"),
               "shader diagnostic omitted its source identity")) break;
    if (!check(contains(compileCapture, "failed to compile"),
               "shader diagnostic omitted its failure description")) break;

    // A fragment shader must report the same three diagnostic properties.
    DiagnosticCapture fragmentCompileCapture;
    SoDebugError::setHandlerCallback(capture_diagnostic, &fragmentCompileCapture);

    SoGLSLShaderObject brokenFragment(glue->contextid);
    brokenFragment.setShaderType(SoGLShaderObject::FRAGMENT);
    brokenFragment.sourceHint = "broken.frag";
    brokenFragment.load("#version 330 core\n"
                        "out vec4 color;\n"
                        "void main() { color = not_an_identifier; }\n");

    if (!check(!brokenFragment.isLoaded(),
               "invalid fragment shader unexpectedly compiled")) break;
    if (!check(fragmentCompileCapture.warnings > 0,
               "fragment compilation failure was not reported as a warning"))
      break;
    if (!check(contains(fragmentCompileCapture, "fragment shader"),
               "fragment shader diagnostic omitted its stage")) break;
    if (!check(contains(fragmentCompileCapture, "broken.frag"),
               "fragment shader diagnostic omitted its source identity")) break;
    if (!check(contains(fragmentCompileCapture, "failed to compile"),
               "fragment shader diagnostic omitted its failure description"))
      break;

    DiagnosticCapture linkCapture;
    SoDebugError::setHandlerCallback(capture_diagnostic, &linkCapture);

    SoGLSLShaderObject vertexShader(glue->contextid);
    vertexShader.setShaderType(SoGLShaderObject::VERTEX);
    vertexShader.sourceHint = "link.vert";
    vertexShader.load("#version 330 core\n"
                      "out vec3 varying_color;\n"
                      "void main() { varying_color = vec3(1.0); "
                      "gl_Position = vec4(0.0); }\n");

    SoGLSLShaderObject fragmentShader(glue->contextid);
    fragmentShader.setShaderType(SoGLShaderObject::FRAGMENT);
    fragmentShader.sourceHint = "link.frag";
    // Call a function that is never defined in any stage.  This compiles
    // cleanly but is guaranteed to fail linking (undefined reference) on
    // every conforming GLSL linker, independent of driver varying-conversion
    // behavior.
    fragmentShader.load("#version 330 core\n"
                        "vec4 helper();\n"
                        "out vec4 color;\n"
                        "void main() { color = helper(); }\n");

    if (!check(vertexShader.isLoaded() && fragmentShader.isLoaded(),
               "link test shaders failed to compile")) break;

    {
      SoGLSLShaderProgram program;
      program.addShaderObject(&vertexShader);
      program.addShaderObject(&fragmentShader);
      program.enable(glue);
    }

    if (!check(linkCapture.warnings > 0,
               "program link failure was not reported as a warning")) break;
    if (!check(contains(linkCapture, "failed to link"),
               "program diagnostic omitted its failure description")) break;
    if (!check(contains(linkCapture, "vertex shader=link.vert"),
               "program diagnostic omitted the vertex source stage")) break;
    if (!check(contains(linkCapture, "fragment shader=link.frag"),
               "program diagnostic omitted the fragment source stage")) break;

    result = 0;
  } while (false);

  SoDebugError::setHandlerCallback(previousCallback, previousData);
  cc_glglue_context_reinstate_previous(context);
  cc_glglue_context_destruct(context);
  SoDB::finish();
  return result;
}
