cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(own_dataset)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED )


add_executable(dataset dataset.cpp)
target_link_libraries(dataset "${TORCH_LIBRARIES}")
target_link_libraries(dataset "${OpenCV_LIBS}")
set_property(TARGET dataset PROPERTY CXX_STANDARD 11)
