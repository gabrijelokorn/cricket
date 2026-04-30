# Requirements
cmake_minimum_required(VERSION 4.3)

# Version and project name
project(cricket VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)

# Target names
set(CLIPGEN  clipgen)
set(CLIPCLASS clipclass)

# --- Shared library ---
add_library(common STATIC
    src/common/Wav.cpp
)
target_include_directories(common PUBLIC
    ${CMAKE_SOURCE_DIR}/src/common
)

# Find packages

# Program for generating learning clips
add_executable(${CLIPGEN}
    src/clipgen/main.cpp
)

add_executable(${CLIPCLASS}
    src/clipclass/main.cpp
)

target_link_libraries(${CLIPGEN}  PRIVATE common)
target_link_libraries(${CLIPCLASS} PRIVATE common)