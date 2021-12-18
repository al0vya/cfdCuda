cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

include(CheckLanguage)

check_language(CUDA)
enable_language(CUDA)
	
enable_language(CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

project(cfdCudaProj)

add_executable(cfdCuda)

if (CMAKE_CUDA_COMPILER)
	
	set_property(TARGET cfdCuda PROPERTY CUDA_SEPARABLE_COMPILATION ON)
	
	add_compile_definitions(CUDA THRUST_IGNORE_CUB_VERSION_CHECK)
	
	target_sources(
	    cfdCuda
		PRIVATE
		source/MAIN.cu
		source/finiteDiff/finiteDiff1Dparallel.cu
		source/finiteDiff/finiteDiff1Dserial.cpp
		source/finiteDiff/finiteDiff2Dparallel.cu
		source/initConds/stepFunction1Dparallel.cu
		source/initConds/stepFunction2Dparallel.cu
		source/run/runConvection1Dparallel.cu
		source/run/runConvection1Dserial.cpp
		source/run/runConvection2Dparallel.cu
		source/MatrixCopy.cu
	)
	
endif()

target_include_directories(
     cfdCuda
	 PRIVATE
	 source
	 source/finiteDiff
	 source/initConds
	 source/run
	 )