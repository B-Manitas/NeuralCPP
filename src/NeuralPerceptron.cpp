/**
 * @file NeuralPerceptron.cpp
 * @see include/NeuralPerceptron.hpp for definition.
 * @brief The NeuralPerceptron class.
 *
 * This file contains the implementation of the NeuralPerceptron class.
 *
 * @author Manitas Bahri <https://github.com/B-Manitas>
 * @date 2023/11
 * @license MIT License
 */

// INCLUDES
#include "../include/NeuralPerceptron.hpp"

// ==================================================
// PRIVATE METHODS

cmatrix<float> NeuralPerceptron::__augment_X(const cmatrix<float> &X) const
{
    return cmatrix<float>::merge(X, cmatrix<float>(X.height(), 1, 1), 1);
}

// ==================================================
// METHODS

cmatrix<float> NeuralPerceptron::fit(const cmatrix<float> &X, const cmatrix<float> &y_true, const int &epochs, const float &learning_rate)
{
    // Check if the number of epochs is valid
    if (epochs <= 0)
        throw std::invalid_argument("The number of epochs must be greater than 0");

    // Check if all the labels are either 1 or -1
    if (y_true.find([&](float x)
                    { return x != 1 && x != -1; }) != std::pair<int, int>(-1, -1))
        throw std::invalid_argument("The labels must be either 1 or -1");

    // Augment the samples matrix with a column of ones for the bias
    cmatrix<float> X_augmented = __augment_X(X);

    // Initialize weights if not set
    if (m_weights.is_empty())
        m_weights = cmatrix<float>::randfloat(X_augmented.width(), 1, -2, 2);

    // Initialize the errors vector
    m_errors = std::vector<float>(epochs);

    // Train the model
    for (int iter = 0; iter < epochs; iter++)
    {
        // Update the weights with the ReLU activation function
        m_weights -= learning_rate * NeuralActivation::dW_relu(X_augmented, y_true, m_weights);

        // Print the error
        if (verbose)
        {
            // Compute the error: number of wrong predictions / total number of samples
            m_errors[iter] = (X_augmented.matmul(m_weights) <= 0).sum_all() / X_augmented.height();

            std::cout << "Epoch: " << iter << " "
                      << "Error: " << m_errors[iter] << std::endl;
        }
    }

    return m_weights;
}

cmatrix<float> NeuralPerceptron::predict(const cmatrix<float> &X) const
{
    // Check if the model is trained
    if (m_weights.is_empty())
        throw std::runtime_error("The model must be trained before making predictions");

    // Check if the number of features should be augmented
    cmatrix<float> X_ = X;
    if (X.width() == m_weights.height() + 1)
        X_ = __augment_X(X);

    // Check if the number of features is valid
    if (X.width() != m_weights.height())
        throw std::invalid_argument("The number of features must be equal to the number of weights: " + std::to_string(m_weights.height()));

    return X_.matmul(m_weights) > 0;
}
