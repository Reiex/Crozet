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

# Add external dependencies

if(NOT TARGET scipp)
    set(SCIPP_ADD_EXAMPLES OFF)
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/external/SciPP)
endif()

if(NOT TARGET ruc)
    set(RUC_ADD_EXAMPLES OFF)
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/external/Ruc)
endif()

if(NOT TARGET diskon)
    set(DISKON_ADD_EXAMPLES OFF)
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/external/Diskon)
endif()

if(NOT TARGET PortAudio)
    set(BUILD_SHARED_LIBS ON)
    set(PA_BUILD_SHARED_TESTS OFF)
    set(PA_BUILD_SHARED_EXAMPLES OFF)
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/external/PortAudio)
endif()

# Crozet

add_library(
    crozet SHARED
    ${CMAKE_CURRENT_LIST_DIR}/include/Crozet/Crozet.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/Crozet/CrozetDecl.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/Crozet/CrozetTypes.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/Crozet/Core/Core.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/Crozet/Core/CoreDecl.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/Crozet/Core/CoreTypes.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/Crozet/Core/AudioDevice.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/Crozet/Core/AudioInput.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/Crozet/Core/AudioOutput.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/Crozet/Core/FilterBase.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/Crozet/Core/FilterPlaySpeed.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/Crozet/Core/SoundBase.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/Crozet/Core/SoundBuffer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/Crozet/Core/SoundFile.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/Crozet/Core/SoundSource.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/Crozet/Core/templates/AudioOutput.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/Crozet/Core/templates/SoundBase.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/Crozet/Private/Private.hpp
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
    scipp
    ruc
    diskon
    PortAudio
)

target_include_directories(
    crozet
    PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include
    PUBLIC ${CMAKE_CURRENT_LIST_DIR}/external/SciPP/include
    PUBLIC ${CMAKE_CURRENT_LIST_DIR}/external/Ruc/include
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
    $<TARGET_LINKER_FILE:PortAudio>
)

# Crozet examples

option(CROZET_ADD_EXAMPLES "Add target crozet-examples" ON)

if(CROZET_ADD_EXAMPLES)

    add_executable(
        crozet-examples
        ${CMAKE_CURRENT_LIST_DIR}/examples/main.cpp
    )

    add_dependencies(
        crozet-examples
        crozet
    )

    target_include_directories(
        crozet-examples
        PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include
        PUBLIC ${CMAKE_CURRENT_LIST_DIR}/external/SciPP/include
        PUBLIC ${CMAKE_CURRENT_LIST_DIR}/external/Ruc/include
        PUBLIC ${CMAKE_CURRENT_LIST_DIR}/external/Diskon/include
    )

    target_link_libraries(
        crozet-examples
        crozet
    )

endif()
