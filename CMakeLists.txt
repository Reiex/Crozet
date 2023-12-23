#######################################################################################################################
##! \file
##! \author Pélégrin Marius
##! \copyright The MIT License (MIT)
##! \date 2022-2023
#######################################################################################################################

# Global project definitions

cmake_minimum_required(VERSION 3.21)

project(Crozet)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

option(CROZET_LIBRARY_ONLY "Build only the library")

# Add external dependencies

set(SCIPP_LIBRARY_ONLY ON)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/external/SciPP)

set(DISKON_LIBRARY_ONLY ON)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/external/Diskon)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/external/PortAudio)

# Crozet

add_library(
    crozet SHARED
    ${CMAKE_CURRENT_LIST_DIR}/src/Core/AudioDevice.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/Core/AudioOutput.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/Core/AudioInput.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/Core/SoundSource.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/Core/SoundBase.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/Core/SoundFile.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/Core/SoundBuffer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/Core/FilterBase.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/Core/FilterPlaySpeed.cpp
)

add_dependencies(
    crozet
    diskon
    PortAudio
)

target_include_directories(
    crozet
    PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include
    PUBLIC ${CMAKE_CURRENT_LIST_DIR}/external/SciPP/include
    PUBLIC ${CMAKE_CURRENT_LIST_DIR}/external/Diskon/include
    PUBLIC ${CMAKE_CURRENT_LIST_DIR}/external/PortAudio/include
)

target_compile_definitions(
    crozet
    PUBLIC CROZET_EXPORTS
)

target_link_libraries(
    crozet
    diskon
    PortAudio
)

# Crozet examples

if(NOT CROZET_LIBRARY_ONLY)

    add_executable(
        crozet-examples
        ${CMAKE_CURRENT_LIST_DIR}/examples/main.cpp
    )

    add_dependencies(
        crozet-examples
        crozet
        diskon
        PortAudio
    )

    target_include_directories(
        crozet-examples
        PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include
        PUBLIC ${CMAKE_CURRENT_LIST_DIR}/external/SciPP/include
        PUBLIC ${CMAKE_CURRENT_LIST_DIR}/external/Diskon/include
    )

    target_link_libraries(
        crozet-examples
        crozet
        diskon
        PortAudio
    )

endif()