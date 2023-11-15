cmake_minimum_required(VERSION 3.15)

project(NeuralCPP VERSION 1.0 LANGUAGES CXX)

# Définition de l'exécutable principal
add_executable(main main.cpp src/model/LinearRegression.cpp src/NeuralLoss.cpp)

# Configuration des tests
include(CTest)

# Ajout de la commande de test
add_test(NAME NeuralTest COMMAND neural_test)

# Définition de l'exécutable des tests
add_executable(neural_test test/NeuralLossTest.cpp src/model/LinearRegression.cpp src/NeuralLoss.cpp)

# Liaison des bibliothèques nécessaires aux tests
target_link_libraries(neural_test gtest)

# Ajout des drapeaux de compilation pour les tests
target_link_libraries(neural_test gtest pthread)
