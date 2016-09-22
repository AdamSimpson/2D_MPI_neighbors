cmake_minimum_required(VERSION 3.1)
project(stencil)

find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)

set(CMAKE_C_STANDARD 99)

file(GLOB SRC_FILES ${CMAKE_SOURCE_DIR}/Source/stencil.c)
add_executable(2D_stencil ${SRC_FILES})

include_directories(${MPI_INCLUDE_PATH})
include_directories(${CUDA_INCLUDE_DIRS})
target_link_libraries(2D_stencil ${MPI_C_LIBRARIES})
target_link_libraries(2D_stencil m)
target_link_libraries(2D_stencil ${CUDA_LIBRARIES})

install(TARGETS 2D_stencil DESTINATION bin)
