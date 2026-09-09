cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 20)

project(nam-cpu-trainer VERSION 0.0.1)

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
	message(STATUS "Enabling /arch:AVX2")		 
else()
	add_compile_options(-march=x86-64-v3)
	message(STATUS "Enabling -march=x86-64-v3")
endif()

set(BUILD_NAMCORE OFF CACHE BOOL "") 

add_subdirectory(deps/cpugrad cpugrad)

set(SOURCES
	NeuralCpuTrainer.cpp
	NAM.h)

add_executable(nam-cpu-trainer ${SOURCES})

target_compile_definitions(nam-cpu-trainer PRIVATE NCT_VERSION_STRING="${PROJECT_VERSION}")

target_include_directories(nam-cpu-trainer PUBLIC deps/cpugrad)
target_include_directories(nam-cpu-trainer PUBLIC deps/cpugrad/NeuralAudio/NeuralAudio)
target_include_directories(nam-cpu-trainer PUBLIC deps/dr_libs)
target_include_directories(nam-cpu-trainer SYSTEM PRIVATE deps/argparse/include/argparse)
target_link_libraries(nam-cpu-trainer PUBLIC cpugrad)

source_group(nam-cpu-trainer ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

