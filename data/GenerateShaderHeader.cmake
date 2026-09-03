if(NOT DEFINED INPUT_FILE OR NOT DEFINED OUTPUT_FILE OR
   NOT DEFINED DEPFILE OR NOT DEFINED SOURCE_ROOT OR
   NOT DEFINED TEMPLATE_FILE OR NOT DEFINED COIN_HEADER_DEF OR
   NOT DEFINED COIN_TEXTVAR_NAME)
  message(FATAL_ERROR
    "GenerateShaderHeader.cmake requires input, output, depfile, source root, template, guard, and symbol arguments")
endif()

function(coin_next_source_id _result)
  get_property(_next GLOBAL PROPERTY COIN_SHADER_NEXT_SOURCE_ID)
  if(NOT _next)
    set(_next 0)
  endif()
  math(EXPR _next "${_next} + 1")
  set_property(GLOBAL PROPERTY COIN_SHADER_NEXT_SOURCE_ID "${_next}")
  set(${_result} "${_next}" PARENT_SCOPE)
endfunction()

function(coin_mask_block_comments _source _result)
  set(_masked "${_source}")
  string(LENGTH "${_source}" _source_length)
  set(_search_offset 0)

  while(_search_offset LESS _source_length)
    string(SUBSTRING "${_source}" ${_search_offset} -1 _tail)
    string(FIND "${_tail}" "/*" _start_relative)
    if(_start_relative EQUAL -1)
      break()
    endif()
    math(EXPR _start "${_search_offset} + ${_start_relative}")
    math(EXPR _after_start "${_start} + 2")

    string(SUBSTRING "${_source}" ${_after_start} -1 _comment_tail)
    string(FIND "${_comment_tail}" "*/" _end_relative)
    if(_end_relative EQUAL -1)
      math(EXPR _end "${_source_length}")
      set(_has_end FALSE)
    else()
      math(EXPR _end "${_after_start} + ${_end_relative} + 2")
      set(_has_end TRUE)
    endif()

    math(EXPR _comment_length "${_end} - ${_start}")
    string(SUBSTRING "${_source}" ${_start} ${_comment_length} _comment)
    string(REGEX REPLACE "[^\n]" " " _masked_comment "${_comment}")
    string(SUBSTRING "${_masked}" 0 ${_start} _prefix)
    string(SUBSTRING "${_masked}" ${_end} -1 _suffix)
    set(_masked "${_prefix}${_masked_comment}${_suffix}")

    if(NOT _has_end)
      break()
    endif()
    set(_search_offset "${_end}")
  endwhile()

  set(${_result} "${_masked}" PARENT_SCOPE)
endfunction()

function(coin_expand_shader _input_file _root_file _stack _result_var _dependencies_var)
  get_filename_component(_input_file "${_input_file}" REALPATH)
  if(NOT EXISTS "${_input_file}")
    message(FATAL_ERROR "GLSL source file not found: ${_input_file}")
  endif()

  list(FIND _stack "${_input_file}" _cycle_index)
  if(NOT _cycle_index EQUAL -1)
    message(FATAL_ERROR
      "GLSL include cycle detected while embedding ${_root_file}: ${_stack};${_input_file}")
  endif()

  file(READ "${_input_file}" _source)
  string(REPLACE "\r\n" "\n" _source "${_source}")
  coin_mask_block_comments("${_source}" _comment_free_source)

  if(_input_file STREQUAL _root_file)
    if(NOT _comment_free_source MATCHES "^[ \t]*#version([ \t]|$)")
      message(FATAL_ERROR
        "GLSL root shader must begin with #version: ${_input_file}")
    endif()
    if(NOT _comment_free_source MATCHES "(^|\n)[ \t]*void[ \t]+main[ \t]*\\(")
      message(FATAL_ERROR
        "GLSL root shader must define main(): ${_input_file}")
    endif()
  else()
    if(_comment_free_source MATCHES "(^|\n)[ \t]*#version([ \t]|$)")
      message(FATAL_ERROR
        "Included GLSL module contains #version: ${_input_file}")
    endif()
    if(_comment_free_source MATCHES "(^|\n)[ \t]*void[ \t]+main[ \t]*\\(")
      message(FATAL_ERROR
        "Included GLSL module must not define main(): ${_input_file}")
    endif()
  endif()

  if(_comment_free_source MATCHES "(^|\n)[ \t]*#[ \t]*include[ \t]*<")
    message(FATAL_ERROR
      "Only quoted relative GLSL includes are supported: ${_input_file}")
  endif()

  set(_dependency_list "${_input_file}")
  if(_input_file STREQUAL _root_file)
    set(_source_id 0)
  else()
    coin_next_source_id(_source_id)
  endif()
  file(RELATIVE_PATH _display_name "${SOURCE_ROOT}" "${_input_file}")

  if(_input_file STREQUAL _root_file)
    # #version must remain the first directive in the generated shader.
    set(_expanded "// coin-source-id: 0 ${_display_name}\n// coin-source: ${_display_name}\n")
  else()
    set(_expanded "#line 1 ${_source_id}\n// coin-source-id: ${_source_id} ${_display_name}\n// coin-source: ${_display_name}\n")
  endif()

  set(_remaining "${_source}")
  set(_line_base 1)
  while(1)
    coin_mask_block_comments("${_remaining}" _searchable_remaining)
    string(REGEX MATCH "(^|\n)[ \t]*#[ \t]*include[ \t]*\"[^\"]+\"" _include
      "${_searchable_remaining}")
    if(NOT _include)
      string(APPEND _expanded "${_remaining}")
      break()
    endif()

    string(FIND "${_searchable_remaining}" "${_include}" _include_position)
    string(SUBSTRING "${_remaining}" 0 ${_include_position} _prefix)
    string(LENGTH "${_include}" _include_length)
    math(EXPR _prefix_length "${_include_position} + ${_include_length}")
    string(SUBSTRING "${_remaining}" ${_prefix_length} -1 _suffix)

    set(_include_directive "${_include}")
    set(_include_has_leading_newline FALSE)
    if(_include MATCHES "^\n")
      set(_include_has_leading_newline TRUE)
      string(REGEX REPLACE "^\n" "" _include_directive
        "${_include_directive}")
    endif()
    string(REGEX REPLACE "^[ \t]*#[ \t]*include[ \t]*\"([^\"]+)\"$"
      "\\1" _include_path "${_include_directive}")
    if(IS_ABSOLUTE "${_include_path}")
      message(FATAL_ERROR
        "GLSL include must be relative to its including file: ${_include_path}")
    endif()

    get_filename_component(_input_directory "${_input_file}" DIRECTORY)
    get_filename_component(_include_file
      "${_input_directory}/${_include_path}" REALPATH)
    if(NOT EXISTS "${_include_file}")
      message(FATAL_ERROR
        "GLSL include not found: ${_include_path} included from ${_display_name}")
    endif()

    string(REGEX MATCHALL "\n" _prefix_newlines "${_prefix}")
    list(LENGTH _prefix_newlines _newline_count)
    math(EXPR _resume_line "${_line_base} + ${_newline_count} + 1")
    if(_include_has_leading_newline)
      math(EXPR _resume_line "${_resume_line} + 1")
    endif()
    string(REGEX REPLACE "^[ \t]*\n" "" _suffix "${_suffix}")

    if(_stack)
      set(_next_stack "${_stack};${_input_file}")
    else()
      set(_next_stack "${_input_file}")
    endif()
    coin_expand_shader("${_include_file}" "${_root_file}" "${_next_stack}"
      _included_source _included_dependencies)
    list(APPEND _dependency_list ${_included_dependencies})
    set(_prefix_separator "")
    if(_include_has_leading_newline)
      set(_prefix_separator "\n")
    endif()
    string(APPEND _expanded
      "${_prefix}${_prefix_separator}${_included_source}\n#line ${_resume_line} ${_source_id}\n")
    set(_remaining "${_suffix}")
    set(_line_base "${_resume_line}")
  endwhile()

  list(REMOVE_DUPLICATES _dependency_list)
  set(${_result_var} "${_expanded}" PARENT_SCOPE)
  set(${_dependencies_var} "${_dependency_list}" PARENT_SCOPE)
endfunction()

function(coin_escape_depfile_path _path _result)
  string(REPLACE "\\" "\\\\" _escaped "${_path}")
  string(REPLACE " " "\\ " _escaped "${_escaped}")
  string(REPLACE "#" "\\#" _escaped "${_escaped}")
  string(REPLACE "$" "$$" _escaped "${_escaped}")
  set(${_result} "${_escaped}" PARENT_SCOPE)
endfunction()

get_filename_component(_root_file "${INPUT_FILE}" REALPATH)
coin_expand_shader("${_root_file}" "${_root_file}" "" _expanded_shader
  _shader_dependencies)

string(REGEX REPLACE "\\\\" "\\\\\\\\" _escaped_shader "${_expanded_shader}")
string(REGEX REPLACE "\"" "\\\\\"" _escaped_shader "${_escaped_shader}")
string(REGEX REPLACE "\r?\n" "\\\\n\"\n  \""
  COIN_STR_SOURCE_CODE "${_escaped_shader}")

get_filename_component(_output_directory "${OUTPUT_FILE}" DIRECTORY)
file(MAKE_DIRECTORY "${_output_directory}")
configure_file("${TEMPLATE_FILE}" "${OUTPUT_FILE}" @ONLY)

coin_escape_depfile_path("${OUTPUT_FILE}" _escaped_output)
file(WRITE "${DEPFILE}" "${_escaped_output}:")
foreach(_dependency IN LISTS _shader_dependencies)
  coin_escape_depfile_path("${_dependency}" _escaped_dependency)
  file(APPEND "${DEPFILE}" " ${_escaped_dependency}")
endforeach()
file(APPEND "${DEPFILE}" "\n")
