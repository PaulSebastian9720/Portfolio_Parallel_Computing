cmake_minimum_required(VERSION 3.20)
project(Preparar CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

add_executable(preparar preparar.cpp)
target_link_libraries(preparar PRIVATE ${OpenCV_LIBS})

add_executable(a_png a_png.cpp)
target_link_libraries(a_png PRIVATE ${OpenCV_LIBS})
