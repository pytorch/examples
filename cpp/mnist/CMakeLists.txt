cmake_minimum_required(VERSION 3.5)
project(mnist)
set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)

option(DOWNLOAD_MNIST "Download the MNIST dataset from the internet" ON)
if (DOWNLOAD_MNIST)
  message(STATUS "Downloading MNIST dataset")
  execute_process(
    COMMAND python ${CMAKE_CURRENT_LIST_DIR}/../tools/download_mnist.py
      -d ${CMAKE_BINARY_DIR}/data
    ERROR_VARIABLE DOWNLOAD_ERROR)
  if (DOWNLOAD_ERROR)
    message(FATAL_ERROR "Error downloading MNIST dataset: ${DOWNLOAD_ERROR}")
  endif()
endif()

add_executable(mnist mnist.cpp)
target_compile_features(mnist PUBLIC cxx_range_for)
target_link_libraries(mnist ${TORCH_LIBRARIES})

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET mnist
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:mnist>)
endif (MSVC)
