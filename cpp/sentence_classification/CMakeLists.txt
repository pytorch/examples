cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(sentence_classification)
set(CMAKE_CXX_STANDARD 14)

find_package(Torch REQUIRED)
include_directories($(CMAKE_CURRENT_SOURCE_DIR)/include)

add_executable(sentence_classification src/datautils.cpp src/sentence_classification.cpp)

target_compile_features(sentence_classification PUBLIC cxx_range_for)
target_link_libraries(sentence_classification ${TORCH_LIBRARIES})

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET sentence_classification
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:sentence_classification>)
endif (MSVC)
