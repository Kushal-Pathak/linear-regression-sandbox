#pragma once
#include <vector>
#include <string>

using Parameters = std::vector<double>;

struct LabeledData
{
    std::vector<std::vector<double>> x;
    std::vector<double> y;
};

struct Model
{
    Parameters parameters;
    std::string name;
    int id;
};

using Models = std::vector<Model>;

struct TrainingSummary{
    int model_id;
    std::string model_name;
    int num_train_samples;
    int num_test_samples;
    int num_features;
    int num_iterations;
    double final_alpha;
    double final_cost;
    double unseen_cost;
    double seen_mae;
    double unseen_mae;
    double seen_r2;
    double unseen_r2;
};
