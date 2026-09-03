if(NOT DEFINED INPUT_FILE OR NOT DEFINED OUTPUT_FILE OR
   NOT DEFINED TEMPLATE_FILE OR NOT DEFINED COIN_HEADER_DEF OR
   NOT DEFINED COIN_TEXTVAR_NAME)
  message(FATAL_ERROR
    "GenerateTextHeader.cmake requires input, output, template, guard, and symbol arguments")
endif()

file(READ "${INPUT_FILE}" _source)
string(REGEX REPLACE "\\\\" "\\\\\\\\" _escaped_source "${_source}")
string(REGEX REPLACE "\"" "\\\\\"" _escaped_source "${_escaped_source}")
string(REGEX REPLACE "\r?\n" "\\\\n\"\n  \""
  COIN_STR_SOURCE_CODE "${_escaped_source}")

get_filename_component(_output_directory "${OUTPUT_FILE}" DIRECTORY)
file(MAKE_DIRECTORY "${_output_directory}")
configure_file("${TEMPLATE_FILE}" "${OUTPUT_FILE}" @ONLY)
