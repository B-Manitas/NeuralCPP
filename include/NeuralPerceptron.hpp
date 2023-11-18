/**
 * @defgroup Perceptron Perceptron
 * @file Perceptron.hpp
 * @see src/model/Perceptron.cpp for implementation.
 * @brief The Perceptron class.
 *
 * This file defines the Perceptron class, which is used to fit a model to a dataset.
 *
 * @see Visit https://en.wikipedia.org/wiki/Perceptron for more information.
 *
 * @author Manitas Bahri <https://github.com/B-Manitas>
 * @date 2023/11
 * @license MIT License
 */

#ifndef PERCEPTRON_HPP
#define PERCEPTRON_HPP

// INCLUDES
#include "NeuralActivation.hpp"

/**
 * @brief This class is used to fit a model to a dataset.
 *
 * @see Visit https://en.wikipedia.org/wiki/Perceptron for more information.
 */
class Perceptron
{
private:
    // STATIC METHODS
    cmatrix<float> __augment_X(const cmatrix<float> &X) const;

public:
    // ATTRIBUTES
    /**
     * @brief The weights of the model. If not set, they are initialized randomly.
     */
    cmatrix<float> m_weights = cmatrix<float>();
    /**
     * @brief The errors of the model. The error is the number of wrong predictions / total number of samples.
     */
    std::vector<float> m_errors = std::vector<float>();
    /**
     * @brief If true, print the error at each iteration. Default: false.
     */
    bool verbose = false;

    // METHODS
    /**
     * @brief Fit the model according to the given training data.
     * It uses the gradient descent algorithm to minimize the loss function and update the
     * weights with the ReLU activation function.
     *
     * @param X The training samples.
     * @param y_true The target values.
     * @param epochs The number of epochs. Default is 1000.
     * @param learning_rate The learning rate. Default is 0.01.
     * @return cmatrix<float> The weights after fitting the model.
     *
     * @throw std::invalid_argument If the number of epochs is not greater than 0.
     * @throw std::invalid_argument If the weights matrix is not of size X.width()x1.
     * @throw std::invalid_argument If the labels are not either 1 or -1.
     *
     * @warning The problem must be a binary classification problem (labels must be either 1 or -1).
     *
     * @note Use the m_errors attribute to get the error after fitting the model.
     * @note Use the m_weights attribute to get the weights after fitting the model.
     * @note Set the verbose attribute to true to print the error after each epoch.
     */

    cmatrix<float> fit(const cmatrix<float> &X, const cmatrix<float> &y_true, const int &epochs = 1000, const float &learning_rate = .01);
    /**
     * @brief Predict using the linear model.
     *
     * @param X The samples matrix.
     * @return cmatrix<float> The predicted values.
     *
     * @throw std::invalid_argument If the model is not fitted.
     * @throw std::invalid_argument If the samples matrix is not of size X.width() x m_weights.height().
     *
     * @note Use the m_weights attribute to get the weights after fitting the model.
     */
    cmatrix<float> predict(const cmatrix<float> &X) const;
};

#endif // PERCEPTRON_HPP
