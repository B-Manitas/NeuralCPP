cmake_minimum_required(VERSION 3.15)

project(NeuralCPP VERSION 1.0 LANGUAGES CXX)
add_executable(main main.cpp src/model/LinearRegression.cpp src/NeuralLoss.cpp)