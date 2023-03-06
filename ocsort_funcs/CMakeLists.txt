cmake_minimum_required(VERSION 3.0)
project(OCSortTracker CXX C)

SET(OPENCV_DIR "" CACHE PATH "Location of libraries")
find_package(OpenCV REQUIRED PATHS ${OPENCV_DIR}/share/OpenCV NO_DEFAULT_PATH)
# find_package(OpenCV 4 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("${CMAKE_SOURCE_DIR}/include")

set(SRCS src/main.cpp src/kalmantracker.cpp src/lapjv.cpp src/ocsort.cpp src/trajectory.cpp)

add_executable(main ${SRCS})

set(DEPS ${DEPS} ${OpenCV_LIBS})
# message(${OpenCV_LIBS})
target_link_libraries(main ${DEPS})



