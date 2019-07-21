cmake_minimum_required(VERSION 2.8)

project(custom-dataset)
set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

add_executable(${PROJECT_NAME} "custom-dataset.cpp")
target_link_libraries(${PROJECT_NAME} "${OpenCV_LIBS}")
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

configure_file("info.txt" "info.txt" COPYONLY)
