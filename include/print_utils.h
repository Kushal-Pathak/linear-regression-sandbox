#pragma once
#include <string>
#include <iostream>
#include <vector>
#include "structs.h"
#include "globals.h"

#define SHOW_VAR(var) std::cout << #var

void print_text(std::string text);

void print_vector(const std::vector<double> &vec);

void print_labeled_data(LabeledData data);

void print_model(Model);

void print_models(std::vector<Model>);

void print_parameters(Parameters);

void show_summary(TrainingSummary summary = TRAINING_SUMMARY);

void show_evaluation(Model model);