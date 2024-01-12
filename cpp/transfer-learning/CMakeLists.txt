cmake_minimum_required(VERSION 3.5)
project(example)

find_package(Torch REQUIRED)
find_package(OpenCV 4.1.0 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(example main.cpp main.h)
add_executable(classify classify.cpp)

target_link_libraries(example ${OpenCV_LIBS})
target_link_libraries(example "${TORCH_LIBRARIES}")
target_link_libraries(classify ${OpenCV_LIBS})
target_link_libraries(classify "${TORCH_LIBRARIES}")

set_property(TARGET classify PROPERTY CXX_STANDARD 17)
set_property(TARGET example PROPERTY CXX_STANDARD 17)
