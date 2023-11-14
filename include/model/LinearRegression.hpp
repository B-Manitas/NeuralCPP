/**
 * @defgroup LinearRegression LinearRegression
 * @file LinearRegression.hpp
 * @see src/model/LinearRegression.cpp for implementation.
 * @brief The LinearRegression class.
 *
 * This file defines the LinearRegression class, which is used to fit a linear model to a dataset.
 * This class inherits from the NeuralModel class.
 *
 * @see Visit https://en.wikipedia.org/wiki/Linear_regression for more information.
 *
 * @author Manitas Bahri <https://github.com/B-Manitas>
 * @date 2023/11
 * @license MIT License
 */

#ifndef LINEARREGRESSION_HPP
#define LINEARREGRESSION_HPP

// INCLUDES
#include "NeuralModel.hpp"
#include "../NeuralLoss.hpp"

/**
 * @brief This class is used to fit a linear model to a dataset.
 * It inherits from the NeuralModel class.
 *
 * @see Visit https://en.wikipedia.org/wiki/Linear_regression for more information.
 */
class LinearRegression : public NeuralModel
{
private:
    // ATTRIBUTES
    std::string m_loss_function = "mse";

public:
    // CONSTRUCTORS
    /**
     * @brief Construct a new LinearRegression object.
     */
    LinearRegression();
    /**
     * @brief Destroy the LinearRegression object.
     */
    ~LinearRegression();

    // SETTERS
    /**
     * @brief Set the loss function.
     *
     * @param loss_function The loss function. It can be either 'mse' or 'mae'.
     *
     * @throw std::invalid_argument If the loss function is not 'mse' or 'mae'.
     */
    void set_loss_function(const std::string &loss_function);

    // METHODS
    /**
     * @brief The linear model. The function is f(X) = x1*w1 + x2*w2 + ... + xn*wn + b.
     *
     * @param X The samples matrix.
     * @param weights The weights matrix.
     * @return cmatrix<float> The predicted values.
     *
     * @throw std::invalid_argument If the weights matrix is not of size X.width()x1.
     *
     * @note Use the m_weights attribute to get the weights after fitting the model.
     */
    static cmatrix<float> model(const cmatrix<float> &X, const cmatrix<float> &weights);
    /**
     * @brief Fit the model according to the given training data.
     * It uses the gradient descent algorithm to minimize the loss function.
     *
     * @param X The training samples.
     * @param y_true The target values.
     * @param epochs The number of epochs. Default is 1000.
     * @param learning_rate The learning rate. Default is 0.01.
     * @param loss_function The loss function. It can be either 'mse' or 'mae'. Default is 'mse'.
     * @return cmatrix<float> The weights after fitting the model.
     *
     * @throw std::invalid_argument If the number of epochs is not greater than 0.
     * @throw std::invalid_argument If the loss function is not 'mse' or 'mae'.
     * @throw std::invalid_argument If the weights matrix is not of size X.width()x1.
     *
     * @note Use the m_errors attribute to get the error after fitting the model.
     * @note Use the m_weights attribute to get the weights after fitting the model.
     * @note Set the verbose attribute to true to print the error after each epoch.
     */
    cmatrix<float> fit(const cmatrix<float> &X, const cmatrix<float> &y_true, const int &epochs = 1000, const float &learning_rate = .01) override;
    /**
     * @brief Predict using the linear model.
     *
     * @param X The samples matrix.
     * @return cmatrix<float> The predicted values.
     *
     * @throw std::invalid_argument If the weights matrix is not of size X.width()x1.
     *
     * @note Use the m_weights attribute to get the weights after fitting the model.
     */
    cmatrix<float> predict(const cmatrix<float> &X) const override;
};

#endif