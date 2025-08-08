#pragma once
#include <iostream>
#include "structs.h"
#include "print_utils.h"
#include "linear_regression.h"

LabeledData generate_dataset(int num_samples, int num_features, double feature_min, double feature_max, double noiseLevel);
