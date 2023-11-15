/**
 * @file NeuralLossTest.cpp
 * @brief The NeuralLoss class test.
 * 
 * This file contains unit tests for the NeuralLoss class.
 * 
 * @author Manitas Bahri
 * @date 2023/11
 * @license MIT License
 */

// INCLUDES
#include <gtest/gtest.h>
#include "../include/NeuralLoss.hpp"

/** @brief Test the mse function. */
TEST(NeuralLossTest, mse)
{
    // TEST 1
    cmatrix<float> y_true = {{0}, {0}, {0}, {0}, {0}};
    cmatrix<float> y_pred = {{1}, {2}, {3}, {4}, {5}};
    float mse = NeuralLoss::mse(y_true, y_pred);
    EXPECT_FLOAT_EQ(mse, 11);

    // TEST 2
    y_true = {{1}, {2}, {3}, {4}, {5}};
    y_pred = {{1}, {2}, {3}, {4}, {5}};
    mse = NeuralLoss::mse(y_true, y_pred);
    EXPECT_FLOAT_EQ(mse, 0);

    // TEST 3
    y_true = {{1}, {2.2}, {1.3}};
    y_pred = {{1}, {2}, {3}};
    mse = NeuralLoss::mse(y_true, y_pred);
    EXPECT_FLOAT_EQ(mse, 0.9766666666666666);

    // TEST 4: INVALID ARGUMENT
    y_true = {{1}, {2}, {3}, {4}, {5}};
    y_pred = {{1}, {2}, {3}, {4}};
    EXPECT_THROW(NeuralLoss::mse(y_true, y_pred), std::invalid_argument);

    // TEST 5: INVALID ARGUMENT
    y_true = {{1}, {2}, {3}, {4}};
    y_pred = {{1}, {2}, {3}, {4}, {5}};
    EXPECT_THROW(NeuralLoss::mse(y_true, y_pred), std::invalid_argument);
}

/** @brief Test the mae function. */
TEST(NeuralLossTest, mae)
{
    // TEST 1
    cmatrix<float> y_true = {{0}, {0}, {0}, {0}, {0}};
    cmatrix<float> y_pred = {{1}, {2}, {3}, {4}, {5}};
    float mae = NeuralLoss::mae(y_true, y_pred);
    EXPECT_FLOAT_EQ(mae, 3);

    // TEST 2
    y_true = {{1}, {2}, {3}, {4}, {5}};
    y_pred = {{1}, {2}, {3}, {4}, {5}};
    mae = NeuralLoss::mae(y_true, y_pred);
    EXPECT_FLOAT_EQ(mae, 0);

    // TEST 3
    y_true = {{1}, {2.2}, {1.3}};
    y_pred = {{1}, {2}, {3}};
    mae = NeuralLoss::mae(y_true, y_pred);
    EXPECT_FLOAT_EQ(mae, 0.6333333333333334);

    // TEST 4: INVALID ARGUMENT
    y_true = {{1}, {2}, {3}, {4}, {5}};
    y_pred = {{1}, {2}, {3}, {4}};
    EXPECT_THROW(NeuralLoss::mae(y_true, y_pred), std::invalid_argument);

    // TEST 5: INVALID ARGUMENT
    y_true = {{1}, {2}, {3}, {4}};
    y_pred = {{1}, {2}, {3}, {4}, {5}};
    EXPECT_THROW(NeuralLoss::mae(y_true, y_pred), std::invalid_argument);
}

GTEST_API_ int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
