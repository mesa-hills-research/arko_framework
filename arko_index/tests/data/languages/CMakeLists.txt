cmake_minimum_required(VERSION 3.20)
project(geometry VERSION 1.2.0 LANGUAGES C CXX)

option(GEOMETRY_TESTS "Build the tests" ON)
set(CMAKE_CXX_STANDARD 20)

# Adds a library with the project's warnings.
function(geometry_library name)
  add_library(${name} ${ARGN})
  target_compile_options(${name} PRIVATE -Wall -Wextra)
  target_include_directories(${name} PUBLIC include)
endfunction()

macro(geometry_option name doc default)
  option(${name} "${doc}" ${default})
  message(STATUS "${name}: ${${name}}")
endmacro()

geometry_library(geometry src/point.cpp src/polygon.cpp)

if(GEOMETRY_TESTS)
  enable_testing()
  add_executable(geometry_tests tests/main.cpp)
  target_link_libraries(geometry_tests PRIVATE geometry)
  add_test(NAME geometry COMMAND geometry_tests)
elseif(WIN32)
  message(WARNING "tests are off")
endif()

foreach(example IN ITEMS hull area)
  add_executable(${example} examples/${example}.cpp)
  target_link_libraries(${example} PRIVATE geometry)
endforeach()

install(TARGETS geometry DESTINATION lib)
