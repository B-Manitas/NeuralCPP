/**
 * @file NeuralLayers.cpp
 * @see include/NeuralLayers.hpp for definition.
 * @brief The NeuralLayers class.
 *
 * This file contains the implementation of the NeuralLayers class.
 *
 * @author Manitas Bahri <https://github.com/B-Manitas>
 * @date 2023/11
 * @license MIT License
 */

// INCLUDES
#include "../include/NeuralLayers.hpp"

// ==================================================
// PRIVATE METHODS

void NeuralLayers::__init_weights(const int &n_features, std::vector<int> &layers_dims, const int &n_output)
{
    // Check if the layers_dims vector is empty
    if (layers_dims.size() == 0)
        throw std::invalid_argument("The layers_dims vector must not be empty");

    // Insert the number of features and the number of outputs in the layers_dims vector
    layers_dims.insert(layers_dims.begin(), n_features);
    layers_dims.push_back(n_output);

    // Initialize the weights
    for (int i = 1; i < layers_dims.size(); i++)
        m_weights["W" + std::to_string(i)] = cmatrix<float>::randfloat(layers_dims[i], layers_dims[i - 1], -1, 1, i);
}

bool NeuralLayers::__valid_layers_dims(const std::vector<int> &layers_dims) const
{
    // Check if the number of neurons in each layer is greater than 0
    for (int i = 1; i < layers_dims.size(); i++)
        if (layers_dims[i] <= 0)
            return false;

    return layers_dims.size() > 1;
}

cmatrix<float> NeuralLayers::__sigmoid(const cmatrix<float> &Z) const
{
    return Z.map([](float x)
                 { return 1 / (1 + std::exp(-x)); });
}

void NeuralLayers::__forward_propagation(const cmatrix<float> &X)
{
    // Initialize the activation for the first layer with the augmented X matrix.
    // Augmented X is the X matrix with a row of ones for the bias.
    m_activations["A0"] = cmatrix<float>::merge(X, cmatrix<float>(1, X.width(), 1), 0);

    // Compute the activation for each layer
    for (int i = 1; i <= m_weights.size(); i++)
    {
        const std::string &dim_str = std::to_string(i);

        // Get the parameters W and the activation A required for the current layer
        const cmatrix<float> &W = m_weights.at("W" + dim_str);
        const cmatrix<float> &A_prev = m_activations.at("A" + std::to_string(i - 1));

        // Compute the activation for the current layer
        const cmatrix<float> &Z = W.matmul(A_prev);
        const cmatrix<float> &A_current = __sigmoid(Z);

        // Store the activation for the current layer
        m_activations["A" + dim_str] = A_current;
    }
}

void NeuralLayers::__back_propagation(const cmatrix<float> &y)
{
    // Initialize variables
    const float m = y.width_t<float>();

    // Compute the gradient for the last layer
    const cmatrix<float> &A_last = m_activations.at("A" + std::to_string(m_activations.size() - 1));
    cmatrix<float> dZ = A_last - y;

    // Compute the gradient for each layer
    for (int i = m_weights.size(); i >= 1; i--)
    {
        const std::string &dim_str = std::to_string(i);
        const std::string &dim_prev_str = std::to_string(i - 1);

        // Get the activation A required for the current layer
        const cmatrix<float> &A_prev = m_activations.at("A" + dim_prev_str);

        // Compute the gradient for the current layer
        const cmatrix<float> &dW = 1 / m * dZ.matmul(A_prev.transpose());

        // Store the gradient for the current layer
        m_gradients["dW" + dim_str] = dW;

        // Compute the gradient for the previous layer
        if (i > 1)
        {
            const cmatrix<float> &W = m_weights.at('W' + dim_str);
            dZ = W.transpose().matmul(dZ) * A_prev * (-1) * (A_prev - (float)1);
        }
    }
}

void NeuralLayers::__gradient_descent(const float &learning_rate)
{
    for (int i = 1; i <= m_weights.size(); i++)
    {
        const std::string &dim_str = std::to_string(i);

        // Get the parameters W required for the current layer
        cmatrix<float> &W = m_weights.at("W" + dim_str);

        // Get the gradients dW required for the current layer
        const cmatrix<float> &dW = m_gradients.at("dW" + dim_str);

        // Update the parameters W for the current layer
        W -= learning_rate * dW;

        // Store the updated parameters W for the current layer
        m_weights["W" + dim_str] = W;
    }
}

// ==================================================
// CONSTRUCTORS

NeuralLayers::NeuralLayers() {}

NeuralLayers::NeuralLayers(const std::vector<int> &layers_dims)
{
    __valid_layers_dims(layers_dims);
    m_layers_dims = layers_dims;
}

NeuralLayers::~NeuralLayers() {}

// ==================================================
// TRAINING METHODS
void NeuralLayers::fit(const cmatrix<float> &X, const cmatrix<float> &y, const int &epochs, const float &learning_rate, const int &verbose)
{
    // Initialize the weights (+1 for the bias)
    __init_weights(X.height() + 1, m_layers_dims, y.height());

    // TODO: Multiclassification
    const cmatrix<cbool> y_true = cmatrix<cbool>(y);

    // Train the model
    for (int i = 0; i < epochs; i++)
    {
        __forward_propagation(X);
        __back_propagation(y);
        __gradient_descent(learning_rate);

        if (verbose != 0 && i % verbose == 0)
        {
            // Predict
            const cmatrix<cbool> &y_pred = predict(X);
            const cmatrix<float> &y_correct = cmatrix<float>(y_pred.eq(y_true));

            // Compute the accuracy
            const float accuracy = y_correct.sum_all() / y_pred.width_t<float>();
            std::cout << i << ". Accuracy: " << accuracy << std::endl;

            // Store the error
            errors.push_back(1 - accuracy);

            // Stop the training if the accuracy is 100%
            if (accuracy == 1)
                break;
        }
    }
}

// ==================================================
// PREDICTION METHODS

cmatrix<cbool> NeuralLayers::predict(const cmatrix<float> &X)
{
    __forward_propagation(X);

    return m_activations.at("A" + std::to_string(m_activations.size() - 1)) > 0.5;
}
