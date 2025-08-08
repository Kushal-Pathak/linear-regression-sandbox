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
    std::cout << "+---------------------------------------------------------------------+\n";
    std::cout << "|                             HELP MENU                               |\n";
    std::cout << "+---------------------------------------------------------------------+\n";
    std::cout << "| 1.  help                         : Show this help menu.             |\n";
    std::cout << "| 2.  about                        : Open about section.              |\n";
    std::cout << "| 3.  version                      : Display program version.         |\n";
    std::cout << "| 4.  clear                        : Clear the console screen.        |\n";
    std::cout << "| 5.  quit                         : Exit the program.                |\n";
    std::cout << "| 6.  load data                    : Load generated dataset.          |\n";
    std::cout << "| 7.  load data <filename>         : Load dataset from file.          |\n";
    std::cout << "| 8.  new model <model_name>       : Create a new model.              |\n";
    std::cout << "| 9.  train model                  : Train currently active model.    |\n";
    std::cout << "| 10. eval model                   : Evaluate current model.          |\n";
    std::cout << "| 11. eval model <id>              : Evaluate by ID.                  |\n";
    std::cout << "| 12. save model                   : Save current model to disk.      |\n";
    std::cout << "| 13. show models                  : List all saved models.           |\n";
    std::cout << "| 14. use model <model_id>         : Select model to use/train.       |\n";
    std::cout << "| 15. current model                : View currently active model.     |\n";
    std::cout << "| 16. input <x1> <x2> ...          : Run inference on inputs.         |\n";
    std::cout << "| 17. rename model <model_id> <new_name>                              |\n";
    std::cout << "|                                  : Change model name.               |\n";
    std::cout << "| 18. delete model <model_id>      : Remove model by ID.              |\n";
    std::cout << "| 19. show trash                   : Show deleted models.             |\n";
    std::cout << "| 20. restore model <model_id>     : Restore model by ID.             |\n";
    std::cout << "| 21. clone model <model_id>       : Create clone of a model.         |\n";
    std::cout << "| 22. gen data <samples> <features> <min> <max> <noise>               |\n";
    std::cout << "|                                  : Create synthetic dataset.        |\n";
    std::cout << "| 23. save data <filename>         : Save generated dataset.          |\n";
    std::cout << "| 24. split data <ratio>           : Splits loaded data into training |\n";
    std::cout << "|                                    and test sets. <ratio> defines   |\n";
    std::cout << "|                                    portion for training;            |\n";
    std::cout << "|                                    rest is test.                    |\n";
    std::cout << "| 25. split data                   : Splits loaded dataset into       |\n";
    std::cout << "|                                    80% training and 20% testing.    |\n";
    std::cout << "| 26. show data                    : Show loaded dataset.             |\n";
    std::cout << "| 27. show data train              : Show training dataset.           |\n";
    std::cout << "| 28. show data test               : Show testing dataset.            |\n";
    std::cout << "| 29. show data gen                : Show generated dataset.          |\n";
    std::cout << "| 30. about model <model_id>       : Show model details by ID.        |\n";
    std::cout << "+---------------------------------------------------------------------+\n";
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
    std::cout << "version 1.0.0\n";
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
    if (filename == "")
    {
        if (GENERATED_DATA.x.empty() || GENERATED_DATA.y.empty())
        {
            std::cout << "Cannot load data, generate data first.\n";
            return;
        }
        LOADED_DATA = GENERATED_DATA;
        std::cout << "Loaded generated data.\n";
    }
    else
    {
        LOADED_DATA = load_CSV(filename);
    }
}

void split_data(std::string split_ratio)
{
    if (LOADED_DATA.x.empty())
    {
        std::cout << "Cannot split data: No data is loaded!\n";
        return;
    }
    try
    {
        double ratio = std::stod(split_ratio);
        int num_samples = LOADED_DATA.x.size();
        int num_training_samples = std::round(num_samples * ratio);
        TRAINING_DATA.x.assign(LOADED_DATA.x.begin(), LOADED_DATA.x.begin() + num_training_samples);
        TRAINING_DATA.y.assign(LOADED_DATA.y.begin(), LOADED_DATA.y.begin() + num_training_samples);
        TESTING_DATA.x.assign(LOADED_DATA.x.begin() + num_training_samples, LOADED_DATA.x.end());
        TESTING_DATA.y.assign(LOADED_DATA.y.begin() + num_training_samples, LOADED_DATA.y.end());
        std::cout << "Data splited into " << ratio * 100 << "% training samples and " << 100 - ratio * 100 << "% testing samples.\n";
    }
    catch (...)
    {
        std::cerr << "Cannot split data: Invalid split ratio!\n";
        return;
    }
}

void show_data(std::string arg)
{
    if (arg == "")
    {
        if (LOADED_DATA.x.empty() || LOADED_DATA.y.empty())
        {
            std::cout << "Cannot show data: Load data first.\n";
            return;
        }
        print_labeled_data(LOADED_DATA);
    }
    else if (arg == "train")
    {
        if (TRAINING_DATA.x.empty() || TRAINING_DATA.y.empty())
        {
            std::cout << "Cannot show data: Load data and split first.\n";
            return;
        }
        print_labeled_data(TRAINING_DATA);
    }
    else if (arg == "test")
    {
        if (TESTING_DATA.x.empty() || TESTING_DATA.y.empty())
        {
            std::cout << "Cannot show data: Load data and split first.\n";
            return;
        }
        print_labeled_data(TESTING_DATA);
    }
    else if (arg == "gen")
    {
        if (GENERATED_DATA.x.empty() || GENERATED_DATA.y.empty())
        {
            std::cout << "Cannot show data: Generate data first.\n";
            return;
        }
        print_labeled_data(GENERATED_DATA);
    }
    else
    {
        std::cout << "Cannot show data: Invalid arg.\n";
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
        std::cout << "Cannot train model, initialize model first.\n";
        return;
    }
    if (TRAINING_DATA.x.empty() || TRAINING_DATA.y.empty())
    {
        std::cout << "Cannot train model, load and split dataset first.\n";
        return;
    }
    CURRENT_MODEL.parameters = gradient_descent(TRAINING_DATA);
    system("cls");
    std::cout << "TRAINING COMPLETED!\n";
    print_model(CURRENT_MODEL);
    if (!GENERATED_PARAMETERS.empty())
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
    if (TESTING_DATA.x.empty() || TESTING_DATA.y.empty())
    {
        std::cout << "Cannot evaluate: Load and split dataset first.\n";
        return;
    }

    if (model_id == "")
    {
        if (!CURRENT_MODEL.id)
        {
            std::cout << "Cannot evaluate: Use a model first.\n";
            return;
        }
        if (TESTING_DATA.x[0].size() != CURRENT_MODEL.parameters.size() - 1)
        {
            std::cout << "Cannot evaluate: Feature size mismatch during evaluation.\n";
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
            if (TESTING_DATA.x[0].size() != model.parameters.size() - 1)
            {
                std::cout << "Cannot evaluate: Feature size mismatch during evaluation.\n";
                return;
            }
        }
        catch (...)
        {
            std::cerr << "Cannot evaluate: Invalid model id.\n";
            return;
        }
    }
    show_evaluation(model);
}

void save_current_model()
{
    if (!CURRENT_MODEL.id)
    {
        std::cout << "Cannot save model, initialize model first.\n";
    }
    else if (CURRENT_MODEL.parameters.empty())
    {
        std::cout << "Cannot save untrained model.\n";
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
    std::cout << "Saved changes.\n";
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
            std::cout << "Cannot use model, model not found.\n";
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
        std::cerr << "Cannot use model, invalid model id.\n";
        return;
    }
}

void show_current_model()
{
    if (CURRENT_MODEL.id)
    {
        print_model(CURRENT_MODEL);
    }
    else
    {
        std::cout << "No models used yet, type new model <model_name>\n";
    }
}

void feed_input(std::vector<std::string> args)
{
    int num_features = args.size() - 1;
    std::vector<double> features;

    if (!CURRENT_MODEL.id)
    {
        std::cout << "Cannot infer, use a model!\n";
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
            std::cout << "Renamed model from " << old_name << " to " << new_name << ".\n";
            if (CURRENT_MODEL.id == id)
            {
                CURRENT_MODEL.name = new_name;
            }
            save_changes_to_file();
        }
    }
    catch (...)
    {
        std::cerr << "Cannot rename model, invalid model id.\n";
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
            std::cout << "Cannot delete model, model not found.\n";
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
        std::cout << "Deleted model with id " << id << ".\n";
        save_changes_to_file();
    }
    catch (...)
    {
        std::cerr << "Cannot delete model, invalid model id.\n";
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
            std::cout << "Cannot restore model, no such model in trash.\n";
            return;
        }
        SAVED_MODELS.push_back(DELETED_MODELS[index]);
        DELETED_MODELS.erase(DELETED_MODELS.begin() + index);
        std::cout << "Restored model with id " << id << ".\n";
        save_changes_to_file();
    }
    catch (...)
    {
        std::cerr << "Cannot restore model, invalid model id.\n";
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
            std::cout << "Cannot clone model, model not found.\n";
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
        std::cout << "Model cloned.\n";
        print_model(CURRENT_MODEL);
        save_changes_to_file();
    }
    catch (...)
    {
        std::cerr << "Cannot clone model, invalid model id.\n";
        return;
    }
}

void gen_data(std::vector<std::string> args)
{
    try
    {
        int num_features = stoi(args[2]);
        int num_samples = stoi(args[3]);
        double min = stod(args[4]);
        double max = stod(args[5]);
        double noiseLevel = stod(args[6]);
        GENERATED_DATA = generate_dataset(num_features, num_samples, min, max, noiseLevel);
        std::cout << "Generated " << num_samples << " data samples with " << num_features << " features between " << min << " and " << max << " with noise level " << noiseLevel << "\n";
    }
    catch (...)
    {
        std::cerr << "Cannot generate data, bad args!\n";
    }
}

void save_data(std::string filename)
{
    if (GENERATED_DATA.x.empty() || GENERATED_DATA.y.empty())
    {
        std::cout << "Cannnot save data, generate data first.\n";
        return;
    }
    save_CSV(GENERATED_DATA, filename);
}

void about_model(std::string model_id)
{
    try
    {
        int id = std::stoi(model_id);
        int index = find_model_index_by_id(id);
        if (index < 0)
        {
            std::cout << "Cannot view model: No such model!\n";
            return;
        }
        Model model = SAVED_MODELS[index];
        std::cout << "+------------------------------------+\n";
        std::cout << "|           MODEL DETAILS            |\n";
        std::cout << "+------------------------------------+\n";
        std::cout << "|" << std::left << std::setw(15) << "Model ID"
                  << ":" << std::setw(20) << model.id << "|\n";
        std::cout << "|" << std::left << std::setw(15) << "Model Name"
                  << ":" << std::setw(20) << model.name << "|\n";
        std::cout << "|" << std::left << std::setw(15) << "No. Parameters"
                  << ":" << std::setw(20) << model.parameters.size() << "|\n";
        std::cout << "+------------------------------------+\n";
        print_vector(model.parameters);
        std::cout << "\n";
    }
    catch (...)
    {
        std::cerr << "Cannot view model: Invalid model id.\n";
        return;
    }
}