cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(interpreter)

find_package(Torch REQUIRED)

# Define our library target
add_library(interpreter SHARED interpreter.cpp)
set(CMAKE_CXX_STANDARD 17)
# Link against LibTorch
target_link_libraries(interpreter "${TORCH_LIBRARIES}")
