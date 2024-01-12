cmake_minimum_required(VERSION 3.5)
project(dist-mnist)

find_package(Torch REQUIRED)

find_package(MPI REQUIRED)

include_directories(SYSTEM ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})

add_executable(dist-mnist dist-mnist.cpp)
target_link_libraries(dist-mnist ${TORCH_LIBRARIES})
target_link_libraries(dist-mnist ${MPI_LIBRARIES})
target_link_libraries(dist-mnist ${CMAKE_PREFIX_PATH}/lib/libc10d.a)

if(MPI_COMPILE_FLAGS)
  set_target_properties(dist-mnist PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(dist-mnist PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()
