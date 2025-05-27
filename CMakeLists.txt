cmake_minimum_required(VERSION 3.16.3)

project(Lesson_5)

set(CMAKE_CXX_STANDARD 17)

# 查找所需的包
find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)
find_package(CURL REQUIRED)
find_package(TIFF REQUIRED)

# 包含目录，确保能找到所有头文件
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${fmt_INCLUDE_DIRS}
    ${CURL_INCLUDE_DIRS}
    ${TIFF_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}
)

# 添加所有源文件
add_executable(main 
    src/main.cpp 
    tasks/detector.cpp 
    tasks/tracker.cpp 
    tasks/PnP.cpp
)

# 链接所需的库
target_link_libraries(main
    ${OpenCV_LIBS}
    fmt::fmt
    ${CURL_LIBRARIES}
    ${TIFF_LIBRARIES}
    pthread              # 添加pthread库支持多线程
)

# 保留原有的链接器标志
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath,/usr/lib/x86_64-linux-gnu")