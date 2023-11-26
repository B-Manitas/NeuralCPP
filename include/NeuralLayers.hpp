/**
 * @defgroup NeuralLayers
 * @file NeuralLayers.hpp
 * @brief NeuralLayers is a class that implements the neural network algorithm.
 * @see src/NeuralLayers.cpp for implementation.
 *
 * @author Manitas Bahri (https://github.com/B-Manitas)
 * @date 2023/11
 * @license MIT License
 */

#ifndef NEURAL_LAYERS_HPP
#define NEURAL_LAYERS_HPP

// INCLUDES
#include <map>

#include "../lib/CMatrix/include/CMatrix.hpp"

class NeuralLayers
{
private:
    // ATTRIBUTES
    std::map<std::string, cmatrix<float>> m_weights = {};
    std::map<std::string, cmatrix<float>> m_activations = {};
    std::map<std::string, cmatrix<float>> m_gradients = {};

    std::vector<int> m_layers_dims = {1};

    // GENERAL METHODS
    /**
     * @brief Initializes the weights for each layer.
     *
     * @param n_features The number of features
     * @param layers_dims The dimensions of each layer
     * @param n_output The number of outputs
     */
    void __init_weights(const int &n_features, std::vector<int> &layers_dims, const int &n_output);
    /**
     * @brief Checks if the dimensions of the layers are valid
     *
     * @param layers_dims The dimensions of each layer. ex: {2, 1} -> 2 neurons in the first layer and 1 neuron in the second layer.
     * @return true if the dimensions of the layers are valid.
     * @return false if the dimensions of the layers are not valid.
     */
    bool __valid_layers_dims(const std::vector<int> &layers_dims) const;
    /**
     * @brief The sigmoid function. It is used as the activation function for the hidden layers.
     *
     * @param Z The matrix to apply the sigmoid function.
     * @return cmatrix<float> The matrix with the sigmoid function applied.
     *
     * @see https://en.wikipedia.org/wiki/Sigmoid_function
     */
    cmatrix<float> __sigmoid(const cmatrix<float> &Z) const;

    // TRAINING METHODS
    /**
     * @brief The forward propagation algorithm. It computes the activation for each layer.
     *
     * @param X The input matrix.
     *
     * @see https://en.wikipedia.org/wiki/Backpropagation#Forward_propagation
     */
    void __forward_propagation(const cmatrix<float> &X);
    /**
     * @brief The back propagation algorithm. It computes the gradients for each layer.
     *
     * @param y The output matrix.
     *
     * @see https://en.wikipedia.org/wiki/Backpropagation#Backpropagation_algorithm
     */
    void __back_propagation(const cmatrix<float> &y);
    /**
     * @brief The gradient descent algorithm. It updates the weights for each layer.
     *
     * @param learning_rate The learning rate.
     *
     * @see https://en.wikipedia.org/wiki/Gradient_descent
     */
    void __gradient_descent(const float &learning_rate);

public:
    // ATTRIBUTES
    /**
     * @brief The errors for each epoch during the training.
     */
    std::vector<float> errors = {};

    // CONSTRUCTORS
    /**
     * @brief Construct a new Neural Layers object containing only one layer with one neuron for binary classification.
     *
     * @note The activation function for the hidden layer is the sigmoid function.
     */
    NeuralLayers();
    /**
     * @brief Construct a new Neural Layers object for binary classification.
     *
     * @param layers_dims The dimensions of each layer. Ex: {2, 1} -> 2 neurons in the first layer and 1 neuron in the second layer.
     *
     * @note The activation function for the hidden layer is the sigmoid function.
     */
    NeuralLayers(const std::vector<int> &layers_dims);
    /**
     * @brief Destroy the Neural Layers object
     */
    ~NeuralLayers();

    // METHODS
    /**
     * @brief Fits the model to the data X and y.
     *
     * @param X The input matrix. Each row represents a sample and each column represents a feature.
     * @param y The output matrix. Must be a row matrix with the same number of rows as X. Must contain only 0 and 1.
     * @param epochs The number of epochs. Default: 1000.
     * @param learning_rate The learning rate. Default: .01.
     * @param verbose The number of epochs between each print of the error. Set to 0 to disable. Default: 100.
     *
     * @note To get the errors for each epoch, use the attribute errors. Ensure that the model is trained before accessing this attribute.
     */
    void fit(const cmatrix<float> &X, const cmatrix<float> &y, const int &epochs = 1000, const float &learning_rate = .01, const int &verbose = 100);

    // PREDICTION METHODS
    /**
     * @brief Predicts the output for the input matrix X.
     *
     * @param X The input matrix.
     * @return cmatrix<cbool> The predicted output matrix.
     */
    cmatrix<cbool> predict(const cmatrix<float> &X);
};

#endif // NEURAL_LAYERS_HPP
