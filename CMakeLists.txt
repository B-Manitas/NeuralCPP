cmake_minimum_required(VERSION 3.15)

project(NeuralCPP VERSION 1.0 LANGUAGES CXX)

file(GLOB_RECURSE SOURCES "src/*.cpp" "src/model/*.cpp")

# Définition de l'exécutable principal
add_executable(main main.cpp ${SOURCES})

# Configuration des tests
include(CTest)

# Ajout de la commande de test
add_test(NAME NeuralTest COMMAND neural_test)

# Définition de l'exécutable des tests
add_executable(neural_test test/NeuralLossTest.cpp ${SOURCES})

# Liaison des bibliothèques nécessaires aux tests
target_link_libraries(neural_test gtest)

# Ajout des drapeaux de compilation pour les tests
target_link_libraries(neural_test gtest pthread)
