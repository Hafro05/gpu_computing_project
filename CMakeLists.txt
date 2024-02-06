cmake_minimum_required(VERSION 3.10)
project(main)

find_package(CUDA QUIET REQUIRED)
find_package(OpenCV REQUIRED)

SET(CUDA_SEPARABLE_COMPILATION ON)

cuda_add_executable(
    main
    main.cpp
    utils.hpp
    utils.cpp
    canny_GPU.cu
    canny_GPU.cuh
    SM_Loader.cu
    SM_Loader.cuh
)

# Linking
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})