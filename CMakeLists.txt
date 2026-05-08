# Requirements
cmake_minimum_required(VERSION 3.5)

# Version and project name
project(cricket VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)
set(NFD_PORTAL ON CACHE BOOL "" FORCE)  # add this BEFORE FetchContent_MakeAvailable

list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})

# Find packages
include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(nlohmann_json)
FetchContent_Declare(
    nfd
    GIT_REPOSITORY https://github.com/btzy/nativefiledialog-extended.git
    GIT_TAG        master
)
FetchContent_MakeAvailable(nfd)
find_package(SndFile CONFIG REQUIRED)
find_package(FFTW3 REQUIRED)
find_package(OpenCV REQUIRED)
# find_package(Torch REQUIRED)

# Target names
set(CLIPGEN  clipgen)
set(CLIPCLASS clipclass)

# --- Shared library ---
add_library(common STATIC
    src/common/Wav.cpp
    src/common/config.cpp
    src/common/gather.cpp
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

target_link_libraries(common PUBLIC 
    nlohmann_json::nlohmann_json
    SndFile::sndfile
    FFTW3::fftw3
    ${OpenCV_LIBS}
    nfd::nfd
)

target_link_libraries(${CLIPGEN}  PRIVATE 
    common
)
    
target_link_libraries(${CLIPCLASS} PRIVATE 
    common
)