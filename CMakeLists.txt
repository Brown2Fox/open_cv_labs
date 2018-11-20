cmake_minimum_required(VERSION 2.8)

project(tracking)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

FILE(GLOB LAB6 lab6.cpp)
FILE(GLOB LAB7 lab7.cpp)
FILE(GLOB LAB8 lab8.cpp)

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
INCLUDE_DIRECTORIES("include/")

add_executable(lab6 ${LAB6})
target_link_libraries(lab6 ${OpenCV_LIBS})

add_executable(lab7 ${LAB7})
target_link_libraries(lab7 ${OpenCV_LIBS})

add_executable(lab8 ${LAB8})
target_link_libraries(lab8 ${OpenCV_LIBS})


#add_custom_command(
#        TARGET lab6 POST_BUILD
#        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/cmake-build-debug/lab6.exe ${CMAKE_SOURCE_DIR}/lab6.exe)
