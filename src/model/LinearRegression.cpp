/**
 * @file LinearRegression.cpp
 * @see src/model/LinearRegression.hpp for definition.
 * @brief The LinearRegression class.
 *
 * This file contains the implementation of the LinearRegression class.
 *
 * @author Manitas Bahri <htpps://github.com/B-Manitas>
 * @date 2023/11
 * @license MIT License
 */

// INCLUDES
#include "../../include/model/LinearRegression.hpp"

// ==================================================
// CONSTRUCTORS

LinearRegression::LinearRegression() {}

LinearRegression::~LinearRegression() {}

// ==================================================
// IMPLEMENT SETTERS

void LinearRegression::set_loss_function(const std::string &loss_function)
{
    // Check if the loss function is valid
    if (loss_function != "mse" && loss_function != "mae")
        throw std::invalid_argument("The loss function must be either 'mse' or 'mae'");

    m_loss_function = loss_function;
}

// ==================================================
// IMPLEMENT STATIC METHODS

cmatrix<float> LinearRegression::model(const cmatrix<float> &X, const cmatrix<float> &weights)
{
    // Check if the weights matrix is of the right size
    if (weights.height() != X.width() || weights.width() != 1)
        throw std::invalid_argument("The weights matrix must be of size " + std::to_string(X.width()) + "x1");

    return X.matmul(weights);
}

// ==================================================
// IMPLEMENT VIRTUAL METHODS

cmatrix<float> LinearRegression::fit(const cmatrix<float> &X, const cmatrix<float> &y_true, const int &epochs, const float &learning_rate)
{
    // Check if the number of epochs is valid
    if (epochs <= 0)
        throw std::invalid_argument("The number of epochs must be greater than 0");

    // Check if the loss function is valid
    std::function<float(cmatrix<float>, cmatrix<float>)> f_loss;
    std::function<cmatrix<float>(cmatrix<float>, cmatrix<float>, cmatrix<float>)> f_loss_grad;

    if (m_loss_function == "mse")
        f_loss = NeuralLoss::mse, f_loss_grad = NeuralLoss::mse_grad;

    else if (m_loss_function == "mae")
        throw std::invalid_argument("The loss function 'mae' is not yet implemented");
        // f_loss = NeuralLoss::mae, f_loss_grad = NeuralLoss::mae_grad;

    else
        throw std::invalid_argument("The loss function must be either 'mse' or 'mae'");

    // Augment the samples matrix with a column of ones for the bias
    cmatrix<float> X_augmented = cmatrix<float>::merge(X, cmatrix<float>(X.height(), 1));

    // Initialize weights if not set
    if (m_weights.is_empty())
        m_weights = cmatrix<float>::randfloat(X.width(), 1, -2, 2);

    // Initialize the errors vector
    m_errors = std::vector<float>(epochs);

    // Train the model
    for (int iter = 0; iter < epochs; iter++)
    {
        m_weights -= learning_rate * f_loss_grad(X, y_true, LinearRegression::model(X, m_weights));
        m_errors[iter] = f_loss(y_true, LinearRegression::model(X, m_weights));

        if (verbose)
            std::cout << "Epoch: " << iter << " "
                      << "Error: " << m_errors[iter] << std::endl;
    }

    return m_weights;
}

cmatrix<float> LinearRegression::predict(const cmatrix<float> &X) const
{
    return model(X, m_weights);
}
