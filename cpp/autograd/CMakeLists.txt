cmake_minimum_required(VERSION 3.5)

project(autograd)
set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)

add_executable(${PROJECT_NAME} "autograd.cpp")
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET ${PROJECT_NAME}
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif (MSVC)
