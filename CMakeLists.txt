cmake_minimum_required(VERSION 3.15)

project(micro_oracle_lm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(micro_oracle_lm main.cpp)

target_include_directories(micro_oracle_lm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
