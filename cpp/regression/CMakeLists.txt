cmake_minimum_required(VERSION 2.8)

project(regression)
set(CMAKE_CXX_STANDARD 11)

find_package(Torch REQUIRED)

add_executable(${PROJECT_NAME} "regression.cpp")
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
