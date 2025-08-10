#include <iostream>
#include "controllers.h"
#include "structs.h"
#include "print_utils.h"
#include "csv.h"
#include <string>
#include "globals.h"
#include "linear_regression.h"
#include "helpers.h"
#include "data_generator.h"
#include <iomanip>
#include <cmath>

void init()
{
    SAVED_MODELS = fetch_models();
    ID_SEED = 0;
    for (Model model : SAVED_MODELS)
    {
        if (model.id > ID_SEED)
        {
            ID_SEED = model.id;
        }
    }
}

void show_logs()
{
    std::cout << "View logs\n";
}

void invalid_command()
{
    std::cout << "Invalid command, type help.\n";
}

void help_program()
{
    std::vector<CommandInfo> command_infos;
    command_infos.push_back({"help", "Show this help menu."});
    command_infos.push_back({"about", "Show program information."});
    command_infos.push_back({"version", "Show program version."});
    command_infos.push_back({"clear", "Clear the console screen."});
    command_infos.push_back({"quit", "Exit the program."});
    command_infos.push_back({"load data <filename>", "Load a dataset from a file."});
    command_infos.push_back({"new model <model_name>", "Create a new model."});
    command_infos.push_back({"train model", "Train the active model."});
    command_infos.push_back({"eval model", "Evaluate the active model."});
    command_infos.push_back({"eval model <model_id>", "Evaluate a model by ID."});
    command_infos.push_back({"save model", "Save the active model."});
    command_infos.push_back({"show models", "List all saved models."});
    command_infos.push_back({"use model <model_id>", "Activate a model by ID."});
    command_infos.push_back({"current model", "Show the active model."});
    command_infos.push_back({"input <x1> <x2> ...", "Run inference with inputs."});
    command_infos.push_back({"rename model <model_id> <new_name>", "Rename a model by ID."});
    command_infos.push_back({"delete model <model_id>", "Delete a model by ID."});
    command_infos.push_back({"show trash", "List deleted models."});
    command_infos.push_back({"restore model <model_id>", "Restore a deleted model."});
    command_infos.push_back({"clone model <model_id>", "Duplicate a model by ID."});
    command_infos.push_back({"gen data <m> <n> <x_min> <x_max> <noise>", "Generate synthetic data (m=samples, n=features)."});
    command_infos.push_back({"save data <filename>", "Save generated data to a file."});
    command_infos.push_back({"split data <ratio>", "Split data into train/test sets. Ratio defines training portion."});
    command_infos.push_back({"split data", "Split data into 80% train / 20% test."});
    command_infos.push_back({"show data", "Show the entire dataset."});
    command_infos.push_back({"show data train", "Show the training dataset."});
    command_infos.push_back({"show data test", "Show the testing dataset."});
    command_infos.push_back({"about model <model_id>", "Show details of a model by ID."});

    std::cout << "+----------------------------------------------------------------------+\n";
    std::cout << "|                            HELP MENU                                 |\n";
    std::cout << "+----------------------------------------------------------------------+\n";
    for (size_t i = 0; i < command_infos.size(); i++)
    {
        std::cout << "|" << std::left << std::setw(3) << i + 1 << ". " << std::setw(65) << command_infos[i].command_name << "|\n";
        std::cout << "|" << std::left << std::setw(5) << "" << std::setw(65) << command_infos[i].command_description << "|\n";
        if (i == command_infos.size() - 1)
        {
            continue;
        }
        std::cout << "|" << std::left << std::setw(70) << "" << "|\n";
    }
    std::cout << "+----------------------------------------------------------------------+\n";
}

void about_program()
{
    std::cout << "+-----------------------------------------------------------------+\n";
    std::cout << "|                 ABOUT LINEAR REGRESSION SANDBOX                 |\n";
    std::cout << "+-----------------------------------------------------------------+\n";
    std::cout << "|A simple console based tool to perform linear regression.        |\n";
    std::cout << "|Allows creation, training, saving, and inference of models.      |\n";
    std::cout << "|Supports loading data from files or generating synthetic data.   |\n";
    std::cout << "|Provides predictions based on input features.                    |\n";
    std::cout << "|Lightweight and easy to use for learning and experimentation.    |\n";
    std::cout << "|Type help to view all available commands.                        |\n";
    std::cout << "|Developed by Kushal Pathak                                       |\n";
    std::cout << "+-----------------------------------------------------------------+\n";
}

void version_program()
{
    std::cout << "version " + version + "\n";
}

void clear_console()
{
    system("cls");
}

void quit_program()
{
    if (CURRENT_MODEL.id && !CURRENT_MODEL.parameters.empty())
    {
        save_current_model();
    }
    std::cout << "Quitting...\n";
    exit(0);
}

void load_data(std::string filename)
{
    try
    {
        DATASET.labeled_data = load_CSV(filename);
        DATASET.generated = false;
    }
    catch (...)
    {
        std::cout << "Failed to load dataset!\n";
        return;
    }
}

void split_data(std::string split_ratio)
{
    LabeledData dataset = DATASET.labeled_data;
    if (dataset.x.empty() || dataset.y.empty())
    {
        std::cout << "Cannot split dataset: Load or generate a dataset first.\n";
        return;
    }
    try
    {
        double ratio = std::stod(split_ratio);
        int num_samples = dataset.x.size();
        int num_training_samples = std::round(num_samples * ratio);
        TRAINING_DATASET.labeled_data.x.assign(dataset.x.begin(), dataset.x.begin() + num_training_samples);
        TRAINING_DATASET.labeled_data.y.assign(dataset.y.begin(), dataset.y.begin() + num_training_samples);
        TESTING_DATASET.labeled_data.x.assign(dataset.x.begin() + num_training_samples, dataset.x.end());
        TESTING_DATASET.labeled_data.y.assign(dataset.y.begin() + num_training_samples, dataset.y.end());

        // track the source of splitted data
        TRAINING_DATASET.generated = DATASET.generated;
        TESTING_DATASET.generated = DATASET.generated;

        std::cout << "Data splited into " << ratio * 100 << "% training samples and " << 100 - ratio * 100 << "% testing samples.\n";
    }
    catch (...)
    {
        std::cerr << "Cannot split dataset: Invalid split ratio!\n";
        return;
    }
}

void show_data(std::string arg)
{
    if (arg == "")
    {
        LabeledData dataset = DATASET.labeled_data;
        if (dataset.x.empty() || dataset.y.empty())
        {
            std::cout << "Cannot show dataset: Load or generate a dataset first.\n";
            return;
        }
        print_labeled_data(dataset);
    }
    else if (arg == "train")
    {
        LabeledData training_data = TRAINING_DATASET.labeled_data;
        if (training_data.x.empty() || training_data.y.empty())
        {
            std::cout << "Cannot show training dataset: Load or generate a dataset, then split it before training.\n";
            return;
        }
        print_labeled_data(training_data);
    }
    else if (arg == "test")
    {
        LabeledData testing_data = TESTING_DATASET.labeled_data;
        if (testing_data.x.empty() || testing_data.y.empty())
        {
            std::cout << "Cannot show testing dataset: Load or generate a dataset, then split it before training.\n";
            return;
        }
        print_labeled_data(testing_data);
    }
    else
    {
        std::cout << "Cannot show dataset: Invalid arg.\n";
    }
}

void new_model(std::string model_name)
{
    if (CURRENT_MODEL.id)
    {
        save_current_model();
    }

    CURRENT_MODEL.name = model_name;
    CURRENT_MODEL.parameters.clear();
    CURRENT_MODEL.id = ++ID_SEED;
    std::cout << "Initialized new model\n";
    print_model(CURRENT_MODEL);
}

void train_model()
{
    if (!CURRENT_MODEL.id)
    {
        std::cout << "Cannot train model: Create or load a model first.\n";
        return;
    }
    LabeledData training_data = TRAINING_DATASET.labeled_data;
    if (training_data.x.empty() || training_data.y.empty())
    {
        std::cout << "Cannot train model: Load or generate a dataset, then split it before training.\n";
        return;
    }
    CURRENT_MODEL.parameters = gradient_descent(training_data);
    // system("cls");
    std::cout << "TRAINING COMPLETED!\n";
    print_model(CURRENT_MODEL);
    if (TRAINING_DATASET.generated)
    {
        std::cout << "TRUE PARAMS: ";
        print_parameters(GENERATED_PARAMETERS);
        GENERATED_PARAMETERS.clear();
    }

    show_summary();
}

void evaluate_model(std::string model_id)
{
    Model model;
    LabeledData testing_data = TESTING_DATASET.labeled_data;
    if (testing_data.x.empty() || testing_data.y.empty())
    {
        std::cout << "Cannot evaluate: Please load and split an evaluation dataset first.\n";
        return;
    }

    if (model_id == "")
    {
        if (!CURRENT_MODEL.id)
        {
            std::cout << "Cannot evaluate: Please use a model first.\n";
            return;
        }
        if (testing_data.x[0].size() != CURRENT_MODEL.parameters.size() - 1)
        {
            std::cout << "Cannot evaluate: Feature size mismatch.\n";
            return;
        }
        model = CURRENT_MODEL;
    }
    else
    {
        try
        {
            int id = std::stoi(model_id);
            int index = find_model_index_by_id(id);
            if (index < 0)
            {
                std::cout << "Cannot evaluate: Model not found.\n";
                return;
            }
            model = SAVED_MODELS[index];
            if (testing_data.x[0].size() != model.parameters.size() - 1)
            {
                std::cout << "Cannot evaluate: Feature size mismatch.\n";
                return;
            }
        }
        catch (...)
        {
            std::cerr << "Cannot evaluate: Invalid model ID.\n";
            return;
        }
    }
    show_evaluation(model);
}

void save_current_model()
{
    if (!CURRENT_MODEL.id)
    {
        std::cout << "Cannot save model: Initialize a model first.\n";
    }
    else if (CURRENT_MODEL.parameters.empty())
    {
        std::cout << "Cannot save model: Model is not trained.\n";
    }
    else
    {
        int index = find_model_index_by_id(CURRENT_MODEL.id);
        if (index > -1)
        {
            SAVED_MODELS[index] = CURRENT_MODEL;
        }
        else
        {
            SAVED_MODELS.push_back(CURRENT_MODEL);
        }
        save_changes_to_file();
    }
}

void save_changes_to_file()
{
    clear_file();
    for (Model model : SAVED_MODELS)
    {
        save_model_to_file(model);
    }
    std::cout << "Changes saved.\n";
    return;
}

void show_models()
{
    print_models(SAVED_MODELS);
}

void use_model(std::string model_id)
{
    try
    {
        int id = std::stoi(model_id);
        int index = find_model_index_by_id(id);
        if (index < 0)
        {
            std::cout << "Cannot use model: Model not found.\n";
            return;
        }
        if (CURRENT_MODEL.id)
        {
            save_current_model();
        }
        CURRENT_MODEL = SAVED_MODELS[index];
        std::cout << "Using model " << CURRENT_MODEL.name << ".\n";
    }
    catch (...)
    {
        std::cerr << "Cannot use model: Invalid model ID.\n";
        return;
    }
}

void show_current_model()
{
    if (CURRENT_MODEL.id)
    {
        about_model(std::to_string(CURRENT_MODEL.id));
    }
    else
    {
        std::cout << "No model in use. Use an existing model or create a new one.\n";
    }
}

void feed_input(std::vector<std::string> args)
{
    int num_features = args.size() - 1;
    std::vector<double> features;

    if (!CURRENT_MODEL.id)
    {
        std::cout << "Cannot infer: Select a model first.\n";
        return;
    }

    if (num_features != CURRENT_MODEL.parameters.size() - 1)
    {
        std::cout << "To infer with " << CURRENT_MODEL.name << " you need " << CURRENT_MODEL.parameters.size() - 1 << " features.\n";
        return;
    }
    try
    {
        for (int i = 1; i < args.size(); i++)
        {
            features.push_back(stod(args[i]));
        }
        double result = inference(CURRENT_MODEL.parameters, features);
        std::cout << "+------------+--------------------+\n";
        std::cout << "|" << std::left << std::setw(12) << "Prediction"
                  << "|" << std::left << std::setw(20) << result << "|\n";
        std::cout << "+------------+--------------------+\n";
    }
    catch (...)
    {
        std::cerr << "Cannot infer, bad args!\n";
        return;
    }
}

void rename_model(std::string model_id, std::string new_name)
{
    try
    {
        int id = std::stoi(model_id);
        int index = find_model_index_by_id(id);
        if (index > -1)
        {
            std::string old_name = SAVED_MODELS[index].name;
            SAVED_MODELS[index].name = new_name;
            std::cout << "Model renamed from " << old_name << " to " << new_name << ".\n";
            if (CURRENT_MODEL.id == id)
            {
                CURRENT_MODEL.name = new_name;
            }
            save_changes_to_file();
        }
    }
    catch (...)
    {
        std::cerr << "Cannot rename model: Invalid model ID.\n";
    }
}

void delete_model(std::string model_id)
{
    try
    {
        int id = std::stoi(model_id);
        int index = find_model_index_by_id(id);
        if (index < 0)
        {
            std::cout << "Cannot delete model: Model not found.\n";
            return;
        }
        if (CURRENT_MODEL.id == id)
        {
            CURRENT_MODEL.id = 0;
            CURRENT_MODEL.name = "";
            CURRENT_MODEL.parameters.clear();
        }
        DELETED_MODELS.push_back(SAVED_MODELS[index]);
        SAVED_MODELS.erase(SAVED_MODELS.begin() + index);
        std::cout << "Model deleted: ID " << id << "\n";
        save_changes_to_file();
    }
    catch (...)
    {
        std::cerr << "Cannot delete model: Invalid model ID.\n";
        return;
    }
}

void show_trash()
{
    print_models(DELETED_MODELS);
}

void restore_model(std::string model_id)
{
    try
    {
        int id = std::stoi(model_id);
        int index = find_model_index_by_id(id, DELETED_MODELS);
        if (index < 0)
        {
            std::cout << "Cannot restore model: No such model in trash.\n";
            return;
        }
        SAVED_MODELS.push_back(DELETED_MODELS[index]);
        DELETED_MODELS.erase(DELETED_MODELS.begin() + index);
        std::cout << "Model restored: ID " << id << "\n";
        save_changes_to_file();
    }
    catch (...)
    {
        std::cerr << "Cannot restore model: Invalid model ID.\n";
        return;
    }
}

void clone_model(std::string model_id)
{
    try
    {
        int id = std::stoi(model_id);
        int index = find_model_index_by_id(id);
        if (index < 0)
        {
            std::cout << "Cannot clone model: Model not found.\n";
            return;
        }
        if (CURRENT_MODEL.id)
        {
            save_current_model();
        }
        CURRENT_MODEL.id = ++ID_SEED;
        CURRENT_MODEL.name = std::string("Clone-") + SAVED_MODELS[index].name;
        CURRENT_MODEL.parameters = SAVED_MODELS[index].parameters;
        SAVED_MODELS.push_back(CURRENT_MODEL);
        std::cout << "Created a clone of model with ID " << id << ".\n";
        print_model(CURRENT_MODEL);
        save_changes_to_file();
    }
    catch (...)
    {
        std::cerr << "Cannot clone model: Invalid model ID.\n";
        return;
    }
}

void gen_data(std::vector<std::string> args)
{
    try
    {
        int num_samples = stoi(args[2]);
        int num_features = stoi(args[3]);
        double min = stod(args[4]);
        double max = stod(args[5]);
        double noiseLevel = stod(args[6]);
        DATASET.labeled_data = generate_dataset(num_samples, num_features, min, max, noiseLevel);
        DATASET.generated = true;
        std::cout << "Generated " << num_samples << " data samples with " << num_features << " features, ranging from " << min << " to " << max << ", with a prediction noise level of " << noiseLevel << "\n";
    }
    catch (...)
    {
        std::cerr << "Cannot generate data: Invalid arguments.\n";
    }
}

void save_data(std::string filename)
{
    LabeledData dataset = DATASET.labeled_data;
    if (dataset.x.empty() || dataset.y.empty())
    {
        std::cout << "Cannot save dataset to file " << filename << ": Generate or load a dataset first.\n";
        return;
    }
    save_CSV(dataset, filename);
}

void about_model(std::string model_id)
{
    try
    {
        int id = std::stoi(model_id);
        int index = find_model_index_by_id(id);
        Model model;
        if (index < 0)
        {
            if (CURRENT_MODEL.id == id)
            {
                model = CURRENT_MODEL;
            }
            else
            {
                std::cout << "Cannot view model: No such model!\n";
                return;
            }
        }
        else
        {
            model = SAVED_MODELS[index];
        }
        print_model(model);
    }
    catch (...)
    {
        std::cerr << "Cannot view model: Invalid model ID.\n";
        return;
    }
}