/**
 * @defgroup NeuralActivation NeuralActivation
 * @file NeuralActivation.hpp
 * @see src/model/NerualActivation.cpp for implementation.
 * @brief The NeuralActivation class.
 *
 * This file defines the activation functions used in neural networks.
 *
 * @see Visit https://en.wikipedia.org/wiki/Activation_function for more information.
 *
 * @author Manitas Bahri <https://github.com/B-Manitas>
 * @date 2023/11
 * @license MIT License
 */

#ifndef NEURALACTIVATION_HPP
#define NEURALACTIVATION_HPP

#include "../lib/CMatrix/include/CMatrix.hpp"

/**
 * @brief This class is used to define the activation functions used in neural networks.
 *
 * @see Visit https://en.wikipedia.org/wiki/Activation_function for more information.
 */
class NeuralActivation
{
public:
    // METHODS
    /**
     * @brief The rectified linear unit (ReLU) activation function.
     * The function is f(Z) = max(0, Z).
     * 
     * @param Z The input matrix.
     *
     * @return The output matrix.
     */
    static cmatrix<float> relu(const cmatrix<float> &Z);

    // UPDATE RULES
    /**
     * @brief The gradient descent update rule for the rectified linear unit (ReLU) activation function.
     * The function is dW = dW + X * y_true if y_pred != y_true.
     * 
     * @param X The training samples.
     * @param y_true The target values.
     * @param weights The weights matrix.
     *
     * @return The updated weights matrix.
     */
    static cmatrix<float> dW_relu(const cmatrix<float> &X, const cmatrix<float> &y_true, const cmatrix<float> &weights);
};

#endif // NEURALACTIVATION_HPP
