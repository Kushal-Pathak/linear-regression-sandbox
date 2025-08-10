#include <iostream>
#include <vector>
#include "linear_algebra.h"
#include "linear_regression.h"
#include "structs.h"
#include <cstdlib>
#include "print_utils.h"
#include <cmath>
#include "globals.h"
#include "helpers.h"
#include <iomanip>

double inference(Parameters params, Parameters features)
{
    if (!features.size() || !params.size())
        return 0;
    double bias = params.back();
    params.pop_back();
    double prediction = 0;
    prediction = dot_product(features, params) + bias;
    return prediction;
}

double cost_fn(Parameters params, LabeledData dataset)
{
    int num_weights = params.size() - 1;
    int num_features = dataset.x[0].size();
    int y_size = dataset.y.size();
    if (num_weights != num_features)
    {
        std::cout << "Cannot calculate cost, number of weights and number of features donot match.\n";
        return -1;
    }
    double cost = 0;
    int m = dataset.x.size();
    for (int i = 0; i < dataset.x.size(); i++)
    {
        auto features = dataset.x[i];
        double y_hat = inference(params, features);
        double y = dataset.y[i];
        double error = (y - y_hat);
        cost += error * error;
    }
    cost /= 2 * m;
    return cost;
}

Parameters get_gradients(Parameters params, LabeledData dataset)
{
    std::vector<double> predictions;
    std::vector<double> errors;
    int m = dataset.x.size(); // total training examples
    int num_features = dataset.x[0].size();
    Parameters gradients(params.size(), 0);

    // calculate predictions & errors
    for (int i = 0; i < m; i++)
    {
        double y_hat = inference(params, dataset.x[i]);
        double y = dataset.y[i];
        double error = y_hat - y;

        // store predictions & errors
        predictions.push_back(y_hat);
        errors.push_back(error);
    }

    // find gradient vector
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < num_features; j++)
        {
            gradients[j] += dataset.x[i][j] * errors[i] / m; // weight gradients
        }
        gradients[num_features] += errors[i] / m; // bias gradient
    }

    return gradients;
}

Parameters gradient_descent(LabeledData dataset)
{
    int num_features = dataset.x[0].size();
    std::vector<double> params(num_features + 1, 0);
    Parameters backupParams = params;
    double alpha = 1;
    double epsilon = 0.0001;
    double significant_decrease = 1;
    double cost = cost_fn(params, dataset);
    double cost_diff = 1 + epsilon;
    double old_cost = cost + cost_diff;
    double very_old_cost = old_cost;
    double initial_cost = cost;
    int initial_iterations = 0;
    int iterations = 0;
    int max_iters = WIDTH;
    double x, y;
    std::string details;
    init_graph(GRAPH, HEIGHT, WIDTH);

    system("cls");
    while (cost_diff > epsilon)
    {
        backupParams = params;
        iterations++;
        Parameters gradients = get_gradients(params, dataset);

        // update params based on new gradients
        gradients = scale_vector(gradients, alpha);
        params = subtract_vectors(params, gradients);

        very_old_cost = old_cost;
        old_cost = cost;
        cost = cost_fn(params, dataset);
        // std::cout << cost;
        cost_diff = old_cost - cost;
        details = "Iterations: " + std::to_string(iterations) + " Cost: " + std::to_string(cost) + " Alpha: " + std::to_string(alpha) + "          " + "\nDecline Rate: " + std::to_string(cost_diff) + "          \nTarget Rate: " + std::to_string(epsilon) + "\n";
        if (cost_diff >= 0)
        {
            // positive difference means descending (good)
            if (cost_diff >= significant_decrease)
            {
                // if cost decreases highly then it's time to pump up learning rate
                alpha *= 1.05;
            }
        }
        else
        {
            // negative difference means ascending (bad)
            params = backupParams;
            cost = old_cost;
            old_cost = very_old_cost;
            cost_diff = old_cost - cost;
            alpha *= 0.3; // decrease alpha because gradient descent is ascending
        }
        try
        {
            x = map_value(iterations, initial_iterations, max_iters, 1, WIDTH - 2);
            y = map_value(cost, 0, initial_cost, 1, HEIGHT - 2);

            if ((cost < std::lround(initial_cost * 0.65) && cost > HEIGHT) || x >= WIDTH - 2)
            {
                initial_cost = cost;
                initial_iterations = iterations;
                max_iters *= 2;
                init_graph(GRAPH, HEIGHT, WIDTH);
            }
            x = map_value(iterations, initial_iterations, max_iters, 1, WIDTH - 2);
            y = map_value(cost, 0, initial_cost, 1, HEIGHT - 2);
            GRAPH[std::lround(y)][std::lround(x)] = '*';
            draw_graph(GRAPH, HEIGHT, WIDTH);
            std::cout << details;
        }
        catch (...)
        {
            std::cout << "Error while graphing.\n";
        }
    }
    TRAINING_SUMMARY.model_id = CURRENT_MODEL.id;
    TRAINING_SUMMARY.model_name = CURRENT_MODEL.name;
    TRAINING_SUMMARY.num_train_samples = dataset.x.size();
    TRAINING_SUMMARY.num_features = num_features;
    TRAINING_SUMMARY.num_iterations = iterations;
    TRAINING_SUMMARY.final_alpha = alpha;
    TRAINING_SUMMARY.final_cost = cost;
    TRAINING_SUMMARY.seen_mae = MAE(params, TRAINING_DATASET.labeled_data);
    TRAINING_SUMMARY.seen_r2 = R_SQUARED(params, TRAINING_DATASET.labeled_data);
    if (TESTING_DATASET.labeled_data.x.empty() || TESTING_DATASET.labeled_data.y.empty())
    {
        TRAINING_SUMMARY.num_test_samples = 0;
        TRAINING_SUMMARY.unseen_cost = 0;
        TRAINING_SUMMARY.unseen_mae = 0;
        TRAINING_SUMMARY.unseen_r2 = 0;
    }
    else
    {
        TRAINING_SUMMARY.num_test_samples = TESTING_DATASET.labeled_data.x.size();
        TRAINING_SUMMARY.unseen_cost = cost_fn(params, TESTING_DATASET.labeled_data);
        TRAINING_SUMMARY.unseen_mae = MAE(params, TESTING_DATASET.labeled_data);
        TRAINING_SUMMARY.unseen_r2 = R_SQUARED(params, TESTING_DATASET.labeled_data);
    }
    return params;
}

double MAE(Parameters params, LabeledData dataset)
{
    int num_weights = params.size() - 1;
    int num_features = dataset.x[0].size();
    int y_size = dataset.y.size();
    if (num_weights != num_features)
    {
        std::cout << "Cannot calculate MAE, number of weights and number of features donot match.\n";
        return -1;
    }
    double mae = 0;
    int m = dataset.x.size();
    for (int i = 0; i < dataset.x.size(); i++)
    {
        auto features = dataset.x[i];
        double y_hat = inference(params, features);
        double y = dataset.y[i];
        double error = (y - y_hat);
        mae += std::fabs(error);
    }
    mae /= m;
    return mae;
}

double R_SQUARED(const Parameters &params, const LabeledData &dataset)
{
    if (dataset.y.empty() || dataset.x.empty())
    {
        std::cout << "Cannot calculate R²: dataset is empty.\n";
        return -1;
    }

    int num_weights = params.size() - 1;
    int num_features = dataset.x[0].size();

    if (num_weights != num_features)
    {
        std::cout << "Cannot calculate R²: number of weights and features do not match.\n";
        return -1;
    }

    double y_mean = 0.0;
    for (double y : dataset.y)
    {
        y_mean += y;
    }
    y_mean /= dataset.y.size();

    double SS_RES = 0.0;
    double SS_TOT = 0.0;

    for (size_t i = 0; i < dataset.x.size(); ++i)
    {
        double y = dataset.y[i];
        double y_hat = inference(params, dataset.x[i]);
        SS_RES += (y - y_hat) * (y - y_hat);
        SS_TOT += (y - y_mean) * (y - y_mean);
    }

    if (SS_TOT == 0.0)
    {
        std::cout << "Cannot calculate R²: all target values are the same.\n";
        return -1;
    }

    return 1.0 - (SS_RES / SS_TOT);
}
