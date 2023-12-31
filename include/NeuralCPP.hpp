/**
 * @defgroup NeuralCPP NeuralCPP
 * @file NeuralCPP.hpp
 * @brief NeuralCPP is a C++ library specialized in the field of machine learning algorithms.
 * @see src/NeuralCPP.cpp for implementation.
 *
 * @author Manitas Bahri (https://github.com/B-Manitas)
 * @date 2023/11
 * @license MIT License
 */

#ifndef NEURALCPP_HPP
#define NEURALCPP_HPP

// INCLUDES
#include "NeuralActivation.hpp"
#include "NeuralLoss.hpp"
#include "NeuralPerceptron.hpp"
#include "NeuralLayers.hpp"

#include <random>

/**
 * @brief NeuralCPP is a C++ library specialized in the field of machine learning algorithms.
 */
class NeuralCPP : public NeuralActivation, public NeuralLoss, public NeuralPerceptron, public NeuralLayers
{
public:
    // METHODS
    /**
     * @brief Create a random dataset.
     *
     * @param X The input matrix. (n_features, n_samples)
     * @param y The output matrix. (1, n_samples)
     * @param n_samples The number of samples.
     * @param n_features The number of features.
     * @param n_classes The number of classes. Default is 2.
     * @param random_state The random state. Default is 0.
     * @return std::tuple<cmatrix<float>, cmatrix<float>> The dataset.
     */
    static void create_dataset(cmatrix<float> &X, cmatrix<float> &y, const int &n_samples, const int &n_features, const int &n_classes = 2, const int &random_state = 0);
};

#endif // NEURALCPP_HPP
