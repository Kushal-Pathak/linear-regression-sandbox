#include <vector>
#include <string>
#include "structs.h"
#include "structs.h"
#include "globals.h"

int ID_SEED;
Model CURRENT_MODEL;
Models SAVED_MODELS;
Models DELETED_MODELS;
Dataset DATASET;
Dataset TRAINING_DATASET;
Dataset TESTING_DATASET;
Parameters GENERATED_PARAMETERS;
TrainingSummary TRAINING_SUMMARY;
int HEIGHT = 10;
int WIDTH = 47;
std::vector<std::string> GRAPH(HEIGHT, std::string(WIDTH, ' '));
std::string version = "2.0.0";
