cmake_minimum_required(VERSION 3.0)
project(transfer-learning)

find_package(Torch REQUIRED)
find_package(OpenCV 4.1.0 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(transfer-learning main.cpp main.h)
add_executable(classify classify.cpp)

target_link_libraries(transfer-learning ${OpenCV_LIBS})
target_link_libraries(transfer-learning "${TORCH_LIBRARIES}")
target_link_libraries(classify ${OpenCV_LIBS})
target_link_libraries(classify "${TORCH_LIBRARIES}")

set_property(TARGET classify PROPERTY CXX_STANDARD 11)
set_property(TARGET transfer-learning PROPERTY CXX_STANDARD 11)
