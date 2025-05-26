# 在CMakeLists.txt中添加

cmake_minimum_required(VERSION 3.16.3)

project(Lesson_5)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)
find_package(CURL REQUIRED)
find_package(TIFF REQUIRED)


include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${fmt_INCLUDE_DIRS}
    ${CURL_INCLUDE_DIRS}
    ${TIFF_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}
)

add_executable(main src/main.cpp tasks/detector.cpp tasks/tracker.cpp)
target_link_libraries(main
    ${OpenCV_LIBS}
    fmt::fmt
    ${CURL_LIBRARIES}
    ${TIFF_LIBRARIES}
)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath,/usr/lib/x86_64-linux-gnu")