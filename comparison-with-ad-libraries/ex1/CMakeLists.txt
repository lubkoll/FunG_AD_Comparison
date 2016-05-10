cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

cmake_policy(SET CMP0015 NEW)

project(fung_ad_comparison_1)

enable_language(C CXX)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y")

include_directories(../../adept-1.1/local/include)
include_directories(../../FADBAD++/FADBAD++)
include_directories(../../trilinos-12.4.2-Source/local/include)
include_directories(../../FunG)
include_directories(../../CppAD/local/include)
include_directories(../../ADOL-C-2.6.1/local/include)

link_directories(../../trilinos-12.4.2-Source/local/lib)
link_directories(../../adept-1.1/local/lib)
link_directories(../../CppAD/local/lib)
link_directories(../../ADOL-C-2.6.1/local/lib64)

add_executable(compare compare.cpp)
target_link_libraries(compare gomp sacado teuchoscore teuchoscomm teuchoskokkoscomm teuchoskokkoscompat teuchosnumerics teuchosparameterlist teuchosremainder adept adolc)
