/**
 * @defgroup NeuralModel NeuralModel
 * @file NeuralModel.hpp
 * @brief The NeuralModel class.
 *
 * This file defines the NeuralModel class, which is used to fit a model to a dataset.
 * This class is inherited by the LinearRegression class.
 *
 * @author Manitas Bahri <https://github.com/B-Manitas>
 * @date 2023/11
 * @license MIT License
 */

#ifndef NEURALMODEL_HPP
#define NEURALMODEL_HPP

// INCLUDES
#include "../NeuralLoss.hpp"
#include "../../lib/CMatrix/include/CMatrix.hpp"

/**
 * @brief This class is used to fit a model to a dataset.
 *
 * Public attributes:
 * - m_weights: The weights coefficients after fitting the model. Ensure that the weights matrix is of size X.height()x1 if custom weights are used.
 * - m_errors: The errors vector after fitting the model.
 * - verbose: Whether to print the error after each epoch. Default is false.
 *
 * @see Visit https://en.wikipedia.org/wiki/artificial_neural_network for more information.
 */
class NeuralModel
{
public:
    // ATTRIBUTES
    /** @brief The weights matrix. */
    cmatrix<float> m_weights = cmatrix<float>();

    /** @brief The errors vector. */
    std::vector<float> m_errors = std::vector<float>();

    /** @brief Whether to print the error after each epoch. */
    bool verbose = false;

    // CONSTRUCTORS
    /**
     * @brief Construct a new NeuralModel object.
     */
    NeuralModel() {}
    /**
     * @brief Destroy the NeuralModel object.
     */
    ~NeuralModel() {}

    // VIRTUAL METHODS
    /**
     * @brief Fit the model according to the given training data.
     *
     * @param X The training samples.
     * @param y_true The target values.
     * @param epochs The number of epochs. Default is 1000.
     * @param learning_rate The learning rate. Default is 0.01.
     * @param loss_function The loss function. It can be either 'mse' or 'mae'. Default is 'mse'.
     * @return cmatrix<float> The weights after fitting the model.
     *
     * @throw std::invalid_argument If the number of epochs is not greater than 0.
     * @throw std::invalid_argument If the weights matrix is not of size X.height()x1.
     *
     * @note Use the m_errors attribute to get the error after fitting the model.
     * @note Use the m_weights attribute to get the weights after fitting the model.
     * @note Set the verbose attribute to true to print the error after each epoch.
     *
     * @warning This method must be implemented in the child class.
     */
    virtual cmatrix<float> fit(const cmatrix<float> &X, const cmatrix<float> &y_true, const int &epochs = 1000, const float &learning_rate = .01) = 0;
    /**
     * @brief Predict using the model.
     *
     * @param X The samples matrix.
     * @return cmatrix<float> The predicted values.
     *
     * @throw std::invalid_argument If the weights matrix is not of size X.height()x1.
     *
     * @note Use the m_weights attribute to get the weights after fitting the model.
     *
     * @warning This method must be implemented in the child class.
     */
    virtual cmatrix<float> predict(const cmatrix<float> &X) const = 0;
};

#endif // NEURALMODEL_HPP