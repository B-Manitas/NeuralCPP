/**
 * @file NeuralLoss.cpp
 * @see src/NeuralLoss.hpp for definition.
 * @brief The NeuralLoss class.
 *
 * This file contains the implementation of the NeuralLoss class.
 *
 * @author Manitas Bahri <htpps://github.com/B-Manitas>
 * @date 2023/11
 * @license MIT License
 */

// INCLUDES
#include "../include/NeuralLoss.hpp"

// ==================================================
// LOSS FUNCTIONS

float NeuralLoss::mse(const cmatrix<float> &y_true, const cmatrix<float> &y_pred)
{
    // Check if the true values matrix is of the right size
    if (y_true.width() != 1)
        throw std::invalid_argument("The true values matrix must be of size y_true.height()x1");

    // Check if the predicted values matrix is of the right size
    if (y_true.height() != y_pred.height())
        throw std::invalid_argument("The prediction values matrix must be of size y_pred.height()x1");

    // Compute the mean squared error
    // MSE: 1/n * sum((y_pred - y_true)^2)
    return 1 / float(y_true.height()) * ((y_pred - y_true) ^ 2).sum_all();
}

// ==================================================
// LOSS GRADIENTS

cmatrix<float> NeuralLoss::mse_grad(const cmatrix<float> &X, const cmatrix<float> &y_true, const cmatrix<float> &y_pred)
{
    // Check if the true values matrix is of the right size
    if (y_true.width() != 1)
        throw std::invalid_argument("The true values matrix must be of size y_true.height()x1");

    // Check if the predicted values matrix is of the right size
    if (y_true.height() != y_pred.height())
        throw std::invalid_argument("The prediction values matrix must be of size y_pred.height()x1");

    // Compute the mean squared error gradient
    // Grad (w): 1/(2n) * X^T * (X * w - y) => 1/(2n) * X^T * (y_pred - y_true) considering y_pred = X * w
    return 1 / (2 * (float)X.height()) * X.transpose().matmul(y_pred - y_true);
}
