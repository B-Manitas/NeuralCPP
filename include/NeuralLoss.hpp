/**
 * @defgroup NeuralLoss NeuralLoss
 * @file NeuralLoss.hpp
 * @see src/NeuralLoss.cpp for implementation.
 * @brief The NeuralLoss class.
 *
 * This file defines the NeuralLoss class, which is used to compute the loss and associated gradients.
 *
 * @see Visit https://en.wikipedia.org/wiki/Loss_function for more information.
 *
 * @author Manitas Bahri <https://github.com/B-Manitas>
 * @date 2023/11
 * @license MIT License
 */

#ifndef NEURALLOSS_HPP
#define NEURALLOSS_HPP

// INCLUDES
#include "../lib/CMatrix/include/CMatrix.hpp"

/**
 * @brief This class is used to compute the loss and associated gradients.
 *
 * @see Visit https://en.wikipedia.org/wiki/Loss_function for more information.
 */
class NeuralLoss
{
private:
    // CHECKS
    /**
     * @brief Check if the true values matrix is of size y_pred.height()x1 and if the predicted values matrix is of size y_pred.height()x1.
     *
     * @param y_true The true values.
     * @param y_pred The predicted values.
     *
     * @throw std::invalid_argument If the true values matrix is not of size y_pred.height()x1.
     * @throw std::invalid_argument If the predicted values matrix is not of size y_pred.height()x1.
     */
    static void __check_valid_y(const cmatrix<float> &y_true, const cmatrix<float> &y_pred);
    /**
     * @brief Check if the samples matrix is of size y_pred.height()xX.width().
     *
     * @param X The samples matrix.
     * @param height The height of the predicted values matrix.
     *
     * @throw std::invalid_argument If the samples matrix is not of size y_pred.height()xX.width().
     */
    static void __check_valid_X(const cmatrix<float> &X, const size_t &height);

public:
    // LOSS FUNCTIONS
    /**
     * @brief Compute the mean squared error (MSE).
     *
     * @param y_true The true values.
     * @param y_pred The predicted values.
     * @return float The mean squared error.
     *
     * @throw std::invalid_argument If the true values matrix is not of size y_pred.height()x1.
     * @throw std::invalid_argument If the predicted values matrix is not of size y_pred.height()x1.
     *
     * @note The MSE is computed as follows: 1/n * sum((y_pred - y_true)^2).
     *
     * @see Visit https://en.wikipedia.org/wiki/Mean_squared_error for more information.
     */
    static float mse(const cmatrix<float> &y_true, const cmatrix<float> &y_pred);
    /**
     * @brief Compute the mean absolute error (MAE).
     *
     * @param y_true The true values.
     * @param y_pred The predicted values.
     * @return float The mean absolute error.
     *
     * @throw std::invalid_argument If the true values matrix is not of size y_pred.height()x1.
     * @throw std::invalid_argument If the predicted values matrix is not of size y_pred.height()x1.
     *
     * @note The MAE is computed as follows: 1/n * sum(|y_pred - y_true|).
     *
     * @see Visit https://en.wikipedia.org/wiki/Mean_absolute_error for more information.
     * @warning NOT IMPLEMENTED YET.
     */
    static float mae(const cmatrix<float> &y_true, const cmatrix<float> &y_pred);

    // LOSS GRADIENTS
    /**
     * @brief Compute the mean squared error gradient.
     *
     * @param X The samples matrix.
     * @param y_true The true values.
     * @param y_pred The predicted values.
     * @return cmatrix<float> The mean squared error gradient.
     *
     * @throw std::invalid_argument If the true values matrix is not of size y_pred.height()x1.
     * @throw std::invalid_argument If the predicted values matrix is not of size y_pred.height()x1.
     *
     * @note The MSE gradient is computed as follows: 1/(2n) * X^T * (y_pred - y_true) considering y_pred = X * w.
     */
    static cmatrix<float> mse_grad(const cmatrix<float> &X, const cmatrix<float> &y_true, const cmatrix<float> &y_pred);
    /**
     * @brief Compute the mean absolute error gradient.
     *
     * @param X The samples matrix.
     * @param y_true The true values.
     * @param y_pred The predicted values.
     * @return cmatrix<float> The mean absolute error gradient.
     *
     * @throw std::invalid_argument If the true values matrix is not of size y_pred.height()x1.
     * @throw std::invalid_argument If the predicted values matrix is not of size y_pred.height()x1.
     *
     * @note The MAE gradient is computed as follows: 1/n * X^T * sign(y_pred - y_true) considering y_pred = X * w.
     * @warning NOT IMPLEMENTED YET.
     */
    static cmatrix<float> mae_grad(const cmatrix<float> &X, const cmatrix<float> &y_true, const cmatrix<float> &y_pred);
};

#endif // NEURALLOSS_HPP