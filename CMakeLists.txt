# Requirements
cmake_minimum_required(VERSION 3.5)

# Version and project name
project(cricket VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)

# Find packages
include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(nlohmann_json)
# find_package(SndFile CONFIG REQUIRED)
# find_package(FFTW3 REQUIRED)
# find_package(OpenCV REQUIRED)
# find_package(Torch REQUIRED)

# Target names
set(CLIPGEN  clipgen)
set(CLIPCLASS clipclass)

# --- Shared library ---
add_library(common STATIC
    src/common/Wav.cpp
    src/common/config.cpp
)
target_include_directories(common PUBLIC
    ${CMAKE_SOURCE_DIR}/src/common
)

# Program for generating learning clips
add_executable(${CLIPGEN}
    src/clipgen/main.cpp
    src/common/json.hpp
    )
    
add_executable(${CLIPCLASS}
    src/clipclass/main.cpp
    src/common/json.hpp
)

target_link_libraries(
    common 
    PRIVATE 
    nlohmann_json::nlohmann_json
)

target_link_libraries(${CLIPGEN}  PRIVATE 
    common
    nlohmann_json::nlohmann_json
)
    
target_link_libraries(${CLIPCLASS} PRIVATE 
    common
    nlohmann_json::nlohmann_json
)