cmake_minimum_required(VERSION 3.10)

project(micrograd VERSION 1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

add_executable(micrograd
    src/Layer.cpp
    src/MLP.cpp
    src/Neuron.cpp
    src/demo.cpp)
