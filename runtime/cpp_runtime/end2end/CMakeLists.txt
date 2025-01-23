cmake_minimum_required(VERSION 3.11)
project(
	tensorRT_cpp_runtime
	LANGUAGES CXX)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/build/cmake" ${CMAKE_MODULE_PATH})

if (NOT TensorRT_DIR)
	set(TensorRT_DIR /usr/src/tensorrt/)
endif()

find_package(TensorRT REQUIRED)
find_package(CUDA REQUIRED)

add_library(utils util.cpp)
target_link_libraries(utils PUBLIC ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES})
target_include_directories(utils PUBLIC ${PROJECT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS} include include/interfaces)

add_executable (cpp_runtime main.cpp)
target_link_libraries(cpp_runtime PUBLIC ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES} utils)
target_include_directories(cpp_runtime PUBLIC ${PROJECT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS} include include/interfaces)
