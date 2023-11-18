/**
 * @file NeuralActivation.cpp
 * @see src/NeuralActivation.hpp for definition.
 * @brief The NeuralActivation class.
 *
 * This file contains the implementation of the NeuralActivation class.
 *
 * @author Manitas Bahri <https://github.com/B-Manitas>
 * @date 2023/11
 * @license MIT License
 */

// INCLUDES
#include "../include/NeuralActivation.hpp"

// ==================================================
// METHODS

cmatrix<float> NeuralActivation::relu(const cmatrix<float> &Z)
{
    return Z.map([](float z)
                 { return std::max(0.0f, z); });
}


// ==================================================
// UPDATE RULES

cmatrix<float> NeuralActivation::dW_relu(const cmatrix<float> &X, const cmatrix<float> &y_true, const cmatrix<float> &weights)
{
    // Compute the predictions
    const cmatrix<float> &y_pred = X.matmul(weights) > 0;
    
    // Get the wrong predictions
    const cmatrix<float> &filter_wrong = (y_pred * y_true) <= 0;
    
    // Update the weights
    cmatrix<float> w = weights;
    for (size_t i = 0; i < X.height(); i++)
        if (filter_wrong.cell(i, 0) == 1)
            w += (y_true.cell(i, 0) * X.rows(i)).transpose();

    return w;
}