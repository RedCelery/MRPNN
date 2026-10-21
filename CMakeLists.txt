cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(MRPNN LANGUAGES CXX CUDA)

enable_language(CUDA)

IF (WIN32)

MESSAGE(STATUS "Is windows")
SET(CMAKE_CXX_FLAGS "/openmp /Ot")

ELSE ()

MESSAGE(STATUS "Is linux")
add_definitions(-DLINUX)

ENDIF ()

Option(GUI OFF)
Option(RTX30XX OFF)
Option(TENSORCORE OFF)
Option(CompileRPNN OFF)
Option(DISABLE_FAST_MATH OFF)

if (GUI)
    add_definitions(-DGUI)
    include(FetchContent)
    set(glew-cmake_BUILD_SHARED OFF CACHE BOOL "" FORCE)
    set(glew-cmake_BUILD_STATIC ON CACHE BOOL "" FORCE)
    FetchContent_Declare(glew URL https://codeload.github.com/Perlmint/glew-cmake/zip/refs/tags/glew-cmake-2.2.0)
    FetchContent_Declare(glfw URL https://codeload.github.com/glfw/glfw/zip/refs/tags/3.4)
    FetchContent_MakeAvailable(glew glfw)
    add_library(GLEW::GLEW ALIAS libglew_static)
    add_subdirectory(Imgui)
    include_directories(${PROJECT_SOURCE_DIR})
endif()

if (CompileRPNN)
    add_definitions(-DCRPNN)
endif()

if (TENSORCORE)
    add_definitions(-DENBALE_TENSOR)
endif()


file(GLOB CORE_HEADERS core/*.h *core/*.hpp core/*.cuh)
file(GLOB CORE_SOURCES core/*.cpp core/*.cu)

source_group("Core Include" FILES ${CORE_HEADERS}) 
source_group("Core Source" FILES ${CORE_SOURCES})

include_directories(${PROJECT_SOURCE_DIR}/core)
  
add_executable(DataGen ${CORE_HEADERS} ${CORE_SOURCES} dataGen.cu)
set_target_properties(DataGen PROPERTIES 
CUDA_SEPARABLE_COMPILATION ON
CXX_STANDARD_REQUIRED ON
CXX_STANDARD 17)
target_compile_definitions(DataGen PRIVATE _HAS_STD_BYTE=0)

add_executable(VolumeRender ${CORE_HEADERS} ${CORE_SOURCES} main.cu)
set_target_properties(VolumeRender PROPERTIES
CUDA_SEPARABLE_COMPILATION ON
CXX_STANDARD_REQUIRED ON
CXX_STANDARD 17)
target_compile_definitions(VolumeRender PRIVATE _HAS_STD_BYTE=0)

add_executable(Test ${CORE_HEADERS} ${CORE_SOURCES} test.cu)
set_target_properties(Test PROPERTIES
CUDA_SEPARABLE_COMPILATION ON
CXX_STANDARD_REQUIRED ON
CXX_STANDARD 17)
target_compile_definitions(Test PRIVATE _HAS_STD_BYTE=0)

add_executable(NetworkProbe ${CORE_HEADERS} ${CORE_SOURCES} training/probe.cu)
set_target_properties(NetworkProbe PROPERTIES CUDA_SEPARABLE_COMPILATION ON
    CXX_STANDARD 17 CUDA_STANDARD 17)
target_compile_definitions(NetworkProbe PRIVATE _HAS_STD_BYTE=0 MRPNN_TRACE)
target_compile_options(NetworkProbe PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

set(MRPNN_FITTED_WEIGHT_HEADER "" CACHE FILEPATH "Exported physical-fit CUDA weights")
if (MRPNN_FITTED_WEIGHT_HEADER)
    if (NOT EXISTS "${MRPNN_FITTED_WEIGHT_HEADER}")
        message(FATAL_ERROR "Fitted weight header does not exist")
    endif()
    file(TO_CMAKE_PATH "${MRPNN_FITTED_WEIGHT_HEADER}" FITTED_HEADER_PATH)
    add_executable(FittedNetworkProbe ${CORE_HEADERS} ${CORE_SOURCES} training/probe.cu)
    set_target_properties(FittedNetworkProbe PROPERTIES CUDA_SEPARABLE_COMPILATION ON
        CXX_STANDARD 17 CUDA_STANDARD 17)
    target_compile_definitions(FittedNetworkProbe PRIVATE _HAS_STD_BYTE=0 MRPNN_TRACE
        MRPNN_WEIGHT_HEADER="${FITTED_HEADER_PATH}")
    target_compile_options(FittedNetworkProbe PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    add_executable(FittedRenderCompare ${CORE_HEADERS} ${CORE_SOURCES} training/render_compare.cu)
    set_target_properties(FittedRenderCompare PROPERTIES CUDA_SEPARABLE_COMPILATION ON
        CXX_STANDARD 17 CUDA_STANDARD 17)
    target_compile_definitions(FittedRenderCompare PRIVATE _HAS_STD_BYTE=0
        MRPNN_WEIGHT_HEADER="${FITTED_HEADER_PATH}")
    target_compile_options(FittedRenderCompare PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endif()

add_executable(RenderCompare ${CORE_HEADERS} ${CORE_SOURCES} training/render_compare.cu)
set_target_properties(RenderCompare PROPERTIES CUDA_SEPARABLE_COMPILATION ON
    CXX_STANDARD 17 CUDA_STANDARD 17)
target_compile_definitions(RenderCompare PRIVATE _HAS_STD_BYTE=0)
target_compile_options(RenderCompare PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(DatasetGen ${CORE_HEADERS} ${CORE_SOURCES} training/dataset.cu)
set_target_properties(DatasetGen PROPERTIES CUDA_SEPARABLE_COMPILATION ON
    CXX_STANDARD 17 CUDA_STANDARD 17)
target_compile_definitions(DatasetGen PRIVATE _HAS_STD_BYTE=0)
target_compile_options(DatasetGen PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

if (CMAKE_CUDA_ARCHITECTURES)
    foreach(target VolumeRender Test DataGen)
        target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
        if (NOT DISABLE_FAST_MATH)
            target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
        endif()
    endforeach()
elseif (DISABLE_FAST_MATH)
    if (RTX30XX)
    target_compile_options(VolumeRender PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_86 -maxrregcount=168>)
    target_compile_options(Test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_86 -maxrregcount=168>)
    target_compile_options(DataGen PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_86 -maxrregcount=168>)
    else()
    target_compile_options(VolumeRender PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_75 -maxrregcount=168>)
    target_compile_options(Test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_75 -maxrregcount=168>)
    target_compile_options(DataGen PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_75 -maxrregcount=168>)
    endif()
else()
    if (RTX30XX)
    target_compile_options(VolumeRender PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math -arch=sm_86 -maxrregcount=168>)
    target_compile_options(Test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math -arch=sm_86 -maxrregcount=168>)
    target_compile_options(DataGen PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math -arch=sm_86 -maxrregcount=168>)
    else()
    target_compile_options(VolumeRender PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math -arch=sm_75 -maxrregcount=168>)
    target_compile_options(Test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math -arch=sm_75 -maxrregcount=168>)
    target_compile_options(DataGen PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math -arch=sm_75 -maxrregcount=168>)
    endif()
endif()

if (GUI)
    target_link_libraries(VolumeRender GLEW::GLEW glfw Imgui) 
    add_dependencies(VolumeRender Imgui)
    target_link_libraries(Test GLEW::GLEW glfw Imgui) 
    add_dependencies(Test Imgui)
    target_link_libraries(DataGen GLEW::GLEW glfw Imgui) 
    add_dependencies(DataGen Imgui)
endif()
