#include <string>
#include "print_utils.h"
#include <iomanip>
#include "linear_regression.h"

void print_text(std::string text)
{
    std::cout << text;
}

void print_vector(const std::vector<double> &vec)
{
    std::cout << "[";
    for (int i = 0; i < vec.size(); i++)
    {
        std::cout << vec[i];
        if (i < vec.size() - 1)
            std::cout << ", ";
    }

    std::cout << "]";
}

void print_labeled_data(LabeledData data)
{
    for (size_t i = 0; i < data.x.size(); ++i)
    {
        for (double v : data.x[i])
            std::cout << v << " ";
        std::cout << "=> " << data.y[i] << "\n";
    }
}

void print_model(Model model)
{
    std::cout << "+----+--------------------+----------+\n";
    std::cout << "|" << std::left << std::setw(4) << "ID" << "|" << std::setw(20) << "NAME" << "|" << std::setw(10) << "PARAMS" << "|\n";
    std::cout << "+----+--------------------+----------+\n";
    std::cout << "|" << std::left << std::setw(4) << model.id << "|" << std::setw(20) << model.name << "|" << std::setw(10) << model.parameters.size() << "|\n";
    std::cout << "+----+--------------------+----------+\n";
    print_vector(model.parameters);
    std::cout << "\n";
}

void print_models(Models models)
{
    std::cout << "+-----+-----+------------------------------+\n";
    std::cout << "|" << std::left << std::setw(5) << "SN" << "|" << std::setw(5) << "ID" << "|" << std::setw(30) << "NAME" << "|\n";
    std::cout << "+-----+-----+------------------------------+\n";
    for (int i = 0; i < models.size(); i++)
    {
        std::cout << "|" << std::left << std::setw(5) << i << "|" << std::setw(5) << models[i].id << "|" << std::setw(30) << models[i].name << "|\n";
    }
    std::cout << "+-----+-----+------------------------------+\n";
}

void print_parameters(Parameters parameters)
{
    std::cout << "[";
    for (int i = 0; i < parameters.size(); i++)
    {
        std::cout << parameters[i];
        if (i < parameters.size() - 1)
            std::cout << ", ";
    }

    std::cout << "]\n";
}

void show_summary(TrainingSummary summary)
{
    int id = summary.model_id;
    std::string name = summary.model_name;
    int num_train_samples = summary.num_train_samples;
    int num_test_samples = summary.num_test_samples;
    int num_feat = summary.num_features;
    int num_itr = summary.num_iterations;
    double alpha = summary.final_alpha;
    double seen_mse = summary.final_cost;
    double unseen_mse = summary.unseen_cost;
    double seen_mae = summary.seen_mae;
    double unseen_mae = summary.unseen_mae;
    double seen_rsq = summary.seen_r2;
    double unseen_rsq = summary.unseen_r2;
    double cost_diff = unseen_mse - seen_mse;
    int col_width = 15;
    std::cout << "+-----------------------------------------------+\n";
    std::cout << "|               TRAINING SUMMARY                |\n";
    std::cout << "+-----------------------------------------------+\n";
    std::cout << "|" << std::left << std::setw(25) << "Model ID"
              << ": " << std::setw(20) << id << "|\n";
    std::cout << "|" << std::left << std::setw(25) << "Model Name"
              << ": " << std::setw(20) << name << "|\n";
    std::cout << "|" << std::left << std::setw(25) << "No. training samples"
              << ": " << std::setw(20) << num_train_samples << "|\n";
    std::cout << "|" << std::left << std::setw(25) << "No. testing samples"
              << ": " << std::setw(20) << num_test_samples << "|\n";
    std::cout << "|" << std::left << std::setw(25) << "No. Features"
              << ": " << std::setw(20) << num_feat << "|\n";
    std::cout << "|" << std::left << std::setw(25) << "No. Iterations"
              << ": " << std::setw(20) << num_itr << "|\n";
    std::cout << "|" << std::left << std::setw(25) << "MSE (seen data)"
              << ": " << std::setw(20) << seen_mse << "|\n";
    std::cout << "|" << std::left << std::setw(25) << "MSE (unseen data)"
              << ": " << std::setw(20) << unseen_mse << "|\n";
    std::cout << "|" << std::left << std::setw(25) << "MSE difference"
              << ": " << std::setw(20) << cost_diff << "|\n";
    std::cout << "|" << std::left << std::setw(25) << "MAE (seen data)"
              << ": " << std::setw(20) << seen_mae << "|\n";
    std::cout << "|" << std::left << std::setw(25) << "MAE (unseen data)"
              << ": " << std::setw(20) << unseen_mae << "|\n";
    std::cout << "|" << std::left << std::setw(25) << "R_Squared (seen data)"
              << ": " << std::setw(20) << seen_rsq << "|\n";
    std::cout << "|" << std::left << std::setw(25) << "R_Squared (unseen data)"
              << ": " << std::setw(20) << unseen_rsq << "|\n";
    std::cout << "+-----------------------------------------------+\n";
}

void show_evaluation(Model model)
{
    if (TRAINING_DATASET.labeled_data.x.empty() || TRAINING_DATASET.labeled_data.y.empty())
    {
        std::cout << "Cannot evaluate: Split training data first.\n";
        return;
    }
    double cost_seen = cost_fn(model.parameters, TRAINING_DATASET.labeled_data);
    double cost_unseen = cost_fn(model.parameters, TESTING_DATASET.labeled_data);
    double cost_diff = cost_unseen - cost_seen;
    double mae_seen = MAE(model.parameters, TRAINING_DATASET.labeled_data);
    double mae_unseen = MAE(model.parameters, TESTING_DATASET.labeled_data);
    double rsq_seen = R_SQUARED(model.parameters, TRAINING_DATASET.labeled_data);
    double rsq_unseen = R_SQUARED(model.parameters, TESTING_DATASET.labeled_data);
    std::string name = model.name;
    std::cout << "+-----------------------------------------------+\n";
    std::cout << "|               MODEL EVALUATION                |\n";
    std::cout << "+-----------------------------------------------+\n";
    std::cout << "|" << std::left << std::setw(25) << "Model ID" << ": " << std::setw(20) << model.id << "|\n";
    std::cout << "|" << std::left << std::setw(25) << "Model Name" << ": " << std::setw(20) << name << "|\n";
    std::cout << "|" << std::left << std::setw(25) << "MSE (seen data)" << ": " << std::setw(20) << cost_seen << "|\n";
    std::cout << "|" << std::left << std::setw(25) << "MSE (unseen data)" << ": " << std::setw(20) << cost_unseen << "|\n";
    std::cout << "|" << std::left << std::setw(25) << "MSE difference" << ": " << std::setw(20) << cost_diff << "|\n";
    std::cout << "|" << std::left << std::setw(25) << "MAE (seen data)" << ": " << std::setw(20) << mae_seen << "|\n";
    std::cout << "|" << std::left << std::setw(25) << "MAE (unseen data)" << ": " << std::setw(20) << mae_unseen << "|\n";
    std::cout << "|" << std::left << std::setw(25) << "R_Squared (seen data)" << ": " << std::setw(20) << rsq_seen << "|\n";
    std::cout << "|" << std::left << std::setw(25) << "R_Squared (unseen data)" << ": " << std::setw(20) << rsq_unseen << "|\n";
    std::cout << "|" << std::left << std::setw(25) << "No. seen data" << ": " << std::setw(20) << TRAINING_DATASET.labeled_data.x.size() << "|\n";
    std::cout << "|" << std::left << std::setw(25) << "No. unseen data" << ": " << std::setw(20) << TESTING_DATASET.labeled_data.x.size() << "|\n";
    std::cout << "+-----------------------------------------------+\n";
}