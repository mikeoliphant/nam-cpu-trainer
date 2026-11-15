set(SOURCES
	NamEval.cpp
)

add_executable(nam-eval ${SOURCES})

target_include_directories(nam-eval PUBLIC ../deps/cpugrad/cpugrad)
target_include_directories(nam-eval PUBLIC ../deps/cpugrad/deps/AudioFFT)
target_include_directories(nam-eval SYSTEM PRIVATE ../deps/cpugrad/NeuralAudio/NeuralAudio)
target_include_directories(nam-eval SYSTEM PRIVATE ../deps/cpugrad/deps/NeuralAudio/deps/RTNeural/modules/Eigen)
target_include_directories(nam-eval SYSTEM PRIVATE ../deps/cpugrad/deps/NeuralAudio/deps/RTNeural/modules/json)
target_include_directories(nam-eval SYSTEM PRIVATE ../deps/dr_libs)
target_include_directories(nam-eval SYSTEM PRIVATE ../deps/argparse/include/argparse)
target_link_libraries(nam-eval PUBLIC cpugrad)

source_group(nam-eval ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

