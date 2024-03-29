
# basic project setup
cmake_minimum_required(VERSION 3.1...3.24)
project(
  nsPerfCmpMultiNode
  VERSION 1.0
  LANGUAGES CXX
)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_BUILD_TYPE Debug)

# define library for shared utility code
add_library(ns_utils utils.cpp utils.h)

add_executable(nsStep12 nsStep12.cpp nsStep12.h)
add_executable(nsStep11 nsStep11.cpp nsStep11.h)

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(nsStep12 PUBLIC OpenMP::OpenMP_CXX ns_utils)
    target_link_libraries(nsStep11 PUBLIC OpenMP::OpenMP_CXX ns_utils)

    target_link_libraries(nsStep12 PUBLIC MPI::MPI_CXX)
    target_link_libraries(nsStep11 PUBLIC MPI::MPI_CXX)
endif()

add_compile_definitions(CREATEPLOTS)

if($ENV{BUILDFORTEST})
    add_compile_definitions(TERMONTOL)
endif()
