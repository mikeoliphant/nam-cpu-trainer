cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 20)

project(nam-cpu-trainer VERSION 0.0.3)

if (MSVC)
	add_compile_options(
		"$<$<CONFIG:DEBUG>:/W4>"
		"$<$<CONFIG:RELEASE>:/O2>"
	)
else()
	add_compile_options(
		-Wall 
		# -Wpedantic -Wextra -Wstrict-aliasing -Wunreachable-code -Weffc++ -Wno-unused-parameter
		"$<$<CONFIG:DEBUG>:-Og;-ggdb>"
		"$<$<CONFIG:RELWITHDEBINFO>:-O3>"
		"$<$<CONFIG:RELEASE>:-O3>"
	)
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options(/arch:AVX2 /fp:fast)
    message(STATUS "Enabling Windows MSVC optimization: /arch:AVX2")       
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" AND (CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64" OR "arm64" IN_LIST CMAKE_OSX_ARCHITECTURES))
    set(TARGET_ARCH "apple-latest" CACHE STRING "Target Apple Silicon CPU for optimization (e.g., apple-latest, apple-m1, apple-m2)")
    add_compile_options("-mcpu=${TARGET_ARCH}")
    message(STATUS "Enabling Apple Silicon optimization: -mcpu=${TARGET_ARCH}")
else()
    set(TARGET_ARCH "native" CACHE STRING "Target architecture for optimization (e.g., native, x86-64-v3)")
    add_compile_options("-march=${TARGET_ARCH}")
    message(STATUS "Enabling standard architecture optimization: -march=${TARGET_ARCH}")
endif()

set(BUILD_NAMCORE OFF CACHE BOOL "") 

add_subdirectory(deps/cpugrad cpugrad)

set(SOURCES
	NeuralCpuTrainer.cpp
	NAM.h
	a2json.h)

add_executable(nam-cpu-trainer ${SOURCES})

target_compile_definitions(nam-cpu-trainer PRIVATE NCT_VERSION_STRING="${PROJECT_VERSION}")

target_include_directories(nam-cpu-trainer PUBLIC deps/cpugrad/cpugrad)
target_include_directories(nam-cpu-trainer SYSTEM PRIVATE deps/cpugrad/NeuralAudio/NeuralAudio)
target_include_directories(nam-cpu-trainer SYSTEM PRIVATE deps/cpugrad/deps/NeuralAudio/deps/RTNeural/modules/Eigen)
target_include_directories(nam-cpu-trainer SYSTEM PRIVATE deps/dr_libs)
target_include_directories(nam-cpu-trainer SYSTEM PRIVATE deps/argparse/include/argparse)
target_link_libraries(nam-cpu-trainer PUBLIC cpugrad)

source_group(nam-cpu-trainer ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

