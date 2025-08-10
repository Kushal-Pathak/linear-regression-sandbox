#pragma once
#include <vector>
#include <string>
#include "structs.h"

extern int ID_SEED;
extern Model CURRENT_MODEL;
extern Models SAVED_MODELS;
extern Models DELETED_MODELS;
extern Dataset DATASET;
extern Dataset TRAINING_DATASET;
extern Dataset TESTING_DATASET;
extern Parameters GENERATED_PARAMETERS;
extern TrainingSummary TRAINING_SUMMARY;
extern int HEIGHT, WIDTH;
extern std::vector<std::string> GRAPH;
extern std::string version;