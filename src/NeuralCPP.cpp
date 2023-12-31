/**
 * @file NeuralCPP.cpp
 * @see include/NeuralCPP.hpp for definition.
 * @brief The NeuralCPP class.
 *
 * This file contains the implementation of the NeuralCPP class.
 *
 * @author Manitas Bahri <https://github.com/B-Manitas>
 * @date 2023/11
 * @license MIT License
 */

// INCLUDES
#include "../include/NeuralCPP.hpp"

// ==================================================
// METHODS

void NeuralCPP::create_dataset(cmatrix<float> &X, cmatrix<float> &y, const int &n_samples, const int &n_features, const int &n_classes, const int &random_state)
{
    // Check if the number of samples is valid
    if (n_samples <= 0)
        throw std::invalid_argument("The number of samples must be greater than 0");

    // Check if the number of features is valid
    if (n_features <= 0)
        throw std::invalid_argument("The number of features must be greater than 0");

    // Check if the number of classes is valid
    if (n_classes <= 0)
        throw std::invalid_argument("The number of classes must be greater than 0");

    // Declare the dataset
    X = cmatrix<float>(n_features, n_samples, -1);
    y = cmatrix<float>(1, n_samples, -1);
    float current_class = -1;

    // Initialize the random generator
    srand(random_state);
    std::uniform_real_distribution<float> dist(-1.0f, 0.0f);
    std::mt19937 generator(random_state);

    // Iterate over each sample and feature to randomly generate the dataset
    for (int c = 0; c < n_samples; c++)
    {
        // Choose a random class
        current_class = (float)(rand() % n_classes);

        // Add some noise
        const float rand_lower_bound = (float)(rand() % 100) / 100.0f;
        const float rand_upper_bound = (float)(rand() % 100) / 100.0f;
        const float sign_x = (rand() % 2) ? 1 : -1;
        const float sign_y = (rand() % 2) ? 1 : -1;

        // Update the distribution
        dist = std::uniform_real_distribution<float>(current_class + rand_lower_bound * sign_x,
                                                     current_class + rand_upper_bound * sign_y);

        // Iterate over each feature
        for (int r = 0; r < n_features; r++)
        {
            const float sign = (r % 2) ? 1 : -1;
            X.cell(r, c) = dist(generator) * sign;
            y.cell(0, c) = current_class;
        }
    }
}