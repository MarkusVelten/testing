cmake_minimum_required(VERSION 3.7)

PROJECT(tests)

SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

SET(ALPAKA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../alpaka" CACHE STRING "The location of the alpaka library")
LIST(APPEND CMAKE_MODULE_PATH "${ALPAKA_ROOT}")
list(APPEND CMAKE_PREFIX_PATH "${ALPAKA_ROOT}")

FIND_PACKAGE("alpaka" REQUIRED)
FIND_PACKAGE(DASH-MPI)

INCLUDE("${ALPAKA_ROOT}/cmake/common.cmake")
#INCLUDE("${ALPAKA_ROOT}/cmake/dev.cmake")

ADD_DEFINITIONS("-D_GLIBCXX_USE_C99=1")
INCLUDE_DIRECTORIES(
    ${CMAKE_CURRENT_LIST_DIR}/../include)
IF(CMAKE_VERSION VERSION_LESS 3.7.0)
    INCLUDE_DIRECTORIES(
        ${alpaka_INCLUDE_DIRS})
    ADD_DEFINITIONS(
        ${alpaka_DEFINITIONS})
ENDIF()

#ALPAKA_ADD_EXECUTABLE(
#    0001-array
#    "0001-array.cpp")
#TARGET_LINK_LIBRARIES(
#    0001-array
#    PUBLIC "alpaka")
#
#IF(DASH-MPI_FOUND)
#    ALPAKA_ADD_EXECUTABLE(
#        0002-foreach
#        "0002-foreach.cpp")
#    TARGET_LINK_LIBRARIES(
#        0002-foreach
#        PUBLIC "alpaka;${DASH_LIBRARIES}")
#ENDIF()

find_package(dash-mpi REQUIRED)
set(LIBRARIES ${LIBRARIES} ${DASH_LIBRARIES})

set(LLAMA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../llama/" CACHE STRING "The location of the llama library")
list(APPEND CMAKE_PREFIX_PATH "${LLAMA_ROOT}")
find_package(llama 0.1.0 REQUIRED)
set(INCLUDE_DIRS ${INCLUDE_DIRS} ${llama_INCLUDE_DIR})
set(INCLUDE_DIRS ${INCLUDE_DIRS} ${LLAMA_ROOT}/examples/common)
add_definitions(${llama_DEFINITIONS})

find_package(alpaka REQUIRED)
set(LIBRARIES ${LIBRARIES} ${alpaka_LIBRARIES})
set(INCLUDE_DIRS ${INCLUDE_DIRS} ${alpaka_INCLUDE_DIRS})
set(DEFINITIONS ${DEFINITIONS} ${alpaka_DEFINITIONS})

#set(DEFINITIONS ${DEFINITIONS} ${ALPAKA_DEV_COMPILE_OPTIONS})
# workaround for: NVCC does not incorporate the alpaka defines

if(ALPAKA_ACC_GPU_CUDA_ENABLE OR ALPAKA_ACC_GPU_HIP_ENABLE)
    set(_ALPAKA_COMPILE_DEFINITIONS_CUDA ${_ALPAKA_COMPILE_DEFINITIONS_PUBLIC})
    list_add_prefix("-D" _ALPAKA_COMPILE_DEFINITIONS_CUDA)
    list(APPEND CUDA_NVCC_FLAGS ${_ALPAKA_COMPILE_DEFINITIONS_CUDA})
endif()

include_directories(0003-localmirror ${INCLUDE_DIRS})
alpaka_add_executable(0003-localmirror 0003-localmirror.cpp Dummy.cpp)
target_compile_options(0003-localmirror PRIVATE ${DEFINITIONS})
target_link_libraries(0003-localmirror PUBLIC ${LIBRARIES})
