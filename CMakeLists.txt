cmake_minimum_required(VERSION 3.5)
project(OV_SD_CPP)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-O3 -march=native -Wall)

find_package(OpenVINO REQUIRED)
find_package(OpenCV REQUIRED)
find_package(pybind11 REQUIRED)
find_package( Boost REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${BOOST_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)

add_executable(SD-generate ${PROJECT_SOURCE_DIR}/src/main.cpp)

target_include_directories(SD-generate PRIVATE 
                            ${PROJECT_BINARY_DIR}/src
                            ${OV_INCLUDE_DIR})

target_link_libraries(SD-generate PRIVATE openvino::runtime pybind11::embed ${OV_LIBRARY} ${OpenCV_LIBS} ${Boost_LIBRARIES} )
