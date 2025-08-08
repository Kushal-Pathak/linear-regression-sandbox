#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include "structs.h"

int save_CSV(LabeledData data, std::string filename, std::string location = "./datasets/");

LabeledData load_CSV(std::string filename, std::string location = "./datasets/");

int save_model_to_file(Model, std::string filepath = "./models/stored_models");

Models fetch_models(std::string filepath = "./models/stored_models");

int clear_file(std::string filepath = "./models/stored_models");