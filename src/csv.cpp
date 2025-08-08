
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include "structs.h"

int save_CSV(LabeledData data, std::string filename, std::string location = "./datasets/")
{
    std::string filepath = location + filename;
    std::ofstream file(filepath);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file.\n";
        return 1;
    }

    for (int i = 1; i <= data.x[0].size(); i++)
    {
        file << "x" << i << ",";
    }
    file << "y\n";

    for (int j = 0; j < data.x.size(); j++)
    {
        for (int i = 0; i < data.x[j].size(); i++)
        {
            file << data.x[j][i];
            file << ",";
        }
        file << data.y[j];
        if (j < data.x.size() - 1)
            file << "\n";
    }
    file.close();

    std::cout << "Data written in " << filepath << " successfully.\n";
    return 0;
}

LabeledData load_CSV(std::string filename, std::string location = "./datasets/")
{
    LabeledData data;
    std::string filepath = location + filename;
    std::ifstream file(filepath);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << filepath << "\n";
        return data;
    }

    bool headerSkipped = false;
    std::string line;

    while (std::getline(file, line))
    {
        if (!headerSkipped)
        {
            headerSkipped = true;
            continue;
        }

        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;

        while (std::getline(ss, cell, ','))
        {
            try
            {
                row.push_back(std::stod(cell));
            }
            catch (...)
            {
                std::cerr << "Warning: Non-numeric value encountered, skipping: " << cell << "\n";
            }
        }
        if (!row.empty())
        {
            double y = row.back();
            row.pop_back();
            data.x.push_back(row);
            data.y.push_back(y);
        }
    }
    file.close();
    std::cout << filepath << " read successfully.\n";
    return data;
}

int save_model_to_file(Model model, std::string filepath = "./models/stored_models")
{
    std::ofstream file(filepath, std::ios::app);
    if (!file.is_open())
    {
        std::cerr << "Failed to open or create " << filepath << "\n";
        return 1;
    }
    file << "\n";
    file << model.id << "," << model.name << ",";
    for (int i = 0; i < model.parameters.size(); i++)
    {
        file << model.parameters[i];
        if (i < model.parameters.size() - 1)
            file << ",";
    }
    // file << "\n";
    file.close();
    // std::cout << "Successfully saved the model \"" << model.name << "\" at " << filepath << "\n";
    return 0;
}

Models fetch_models(std::string filepath)
{
    // load all models into memory
    Models models;
    std::ifstream file(filepath);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << filepath << "\n";
        return models;
    }

    std::string line;

    bool headerSkipped = false;

    while (std::getline(file, line))
    {
        if (!headerSkipped)
        {
            headerSkipped = true;
            continue;
        }

        std::stringstream ss(line);
        std::string cell;
        Model model;

        // Get model id
        std::getline(ss, cell, ',');
        try
        {
            model.id = std::stoi(cell);
        }
        catch (...)
        {
            std::cerr << "Warning: Skipping invalid value " << cell << "\n";
        }

        // Get model name
        std::getline(ss, cell, ',');
        model.name = cell;

        while (std::getline(ss, cell, ','))
        {
            try
            {
                model.parameters.push_back(std::stod(cell));
            }
            catch (...)
            {
                std::cerr << "Warning: Skipping invalid value " << cell << "\n";
            }
        }
        models.push_back(model);
    }
    file.close();
    std::cout << "Fetched all models from " << filepath << "\n";
    return models;
}

int clear_file(std::string filepath = "./models/stored_models")
{
    std::ofstream file(filepath);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file.\n";
        return 1;
    }

    file << "ID,NAME,PARAMETERS...";
    file.close();
    return 0;
}