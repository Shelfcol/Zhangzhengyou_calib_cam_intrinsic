cmake_minimum_required(VERSION 3.5)
project(CameraIntrinsicsCalib)

set(CMAKE_BUILD_TYPE "Debug")
message("Build type: " ${CMAKE_BUILD_TYPE})
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-O3 -Wall -std=c++11")

option(USE_OPENCV_2 "choose opencv2.4.13 or opencv3.4.16" ON)

if(USE_OPENCV_2)
    find_package(OpenCV REQUIRED
              PATHS "/usr/local/opencv-2.4.13/share/OpenCV/" NO_DEFAULT_PATH)
    MESSAGE(STATUS "Using OpenCV2")
else()
    find_package(OpenCV 3 REQUIRED
              PATHS "/usr/local/opencv-3.4.16/share/OpenCV/" NO_DEFAULT_PATH)
    MESSAGE(STATUS "Using OpenCV3")
endif()

find_package(Ceres REQUIRED)

include_directories(
 ${PROJECT_SOURCE_DIR}/include
 ${CERES_INCLUDE_DIRS}
)

add_executable(main src/main.cpp src/camera_calib.cpp)
target_link_libraries(main ${OpenCV_LIBS} ${CERES_LIBRARIES})
