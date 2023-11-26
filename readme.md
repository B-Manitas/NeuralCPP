# NeuralCPP: A Neural Network Library in C++

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Active-green.svg)](https://github.com/B-Manitas/CMatrix)

NeuralCPP is a Neural Network Library written in C++. It offers a simple API to create and train neural networks.

## Table of Contents

1. [Installation](#installation)
2. [Example](#example)
3. [Hierarchical Structure](#hierarchical-structure)
4. [Documentation](#documentation)
5. [Libraries Used](#libraries-used)
6. [See Also](#see-also)
7. [License](#license)

## Installation

To add NeuralCPP to your project, follow these steps:

1. Add NeuralCPP as a submodule to your project:

```bash
git submodule add -b main https://github.com/B-Manitas/NeuralCPP.git
git submodule update --init --recursive
```

2. Include the [`NeuralCPP/include/NeuralCPP.hpp`](include/NeuralCPP.hpp) file in your project.

3. Compile your project with the following flags:

```bash
-std=c++11 -fopenmp
```

## Example

```cpp
#include "include/NeuralCPP.hpp"

int main()
{
    // Create the dataset
    cmatrix<float> X, y;
    NeuralCPP::create_dataset(X, y, 100000, 2, 2);

    // Create the neural network model (2 hidden layers with 32 neurons each)
    NeuralLayers nn({32, 32});

    // Train the model
    nn.fit(X, y, 10000, .01, true);

    // Predict the output
    cmatrix<cbool> y_pred = nn.predict(X);

    return 0;
}
```

## Hierarchical Structure

NeuralCPP is structured as follows:

| Class                                                        | Description                                             |
| ------------------------------------------------------------ | ------------------------------------------------------- |
| **include**                                                  |                                                         |
| [`NeuralCPP.hpp`](include/NeuralCPP.hpp)                     | Includes all the other headers.                         |
| [`NeuralLoss.hpp`](include/NeuralLoss.hpp)                   | Defines the loss functions.                             |
| [`NeuralActivation.hpp`](include/model/LinearRegression.hpp) | Defines the activation functions.                       |
| [`NeuralPerceptron.hpp`](include/model/NeuralModel.hpp)      | The Perceptron model.                                   |
| **src**                                                      |                                                         |
|                                                              | This folder contains the implementation of the library. |

## Documentation

For detailed information on how to use CMatrix, consult the [documentation](docs/neuralcpp.pdf).

## Libraries Used

- [CMatrix](https://github.com/B-Manitas/CMatrix): A C++ library for matrix operations. _(Required for compile CMatrix)_
- [OpenMP](https://www.openmp.org/): An API for parallel programming. _(Required for compile CMatrix)_
- [Doxygen](https://www.doxygen.nl): A documentation generator.
- [Matplot++](https://github.com/alandefreitas/matplotplusplus): A C++ Graphics Library for Data Visualization.

## See Also

- [CMatrix](https://github.com/B-Manitas/CMatrix): A C++ library for matrix operations.
- [CDataFrame](https://github.com/B-Manitas/CDataFrame): A C++ DataFrame library for Data Science and Machine Learning projects.

## License

This project is licensed under the MIT License, ensuring its free and open availability to the community.
