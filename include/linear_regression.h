#include <iostream>
#include <vector>
#include "linear_algebra.h"
#include "structs.h"
#include <cstdlib>

double inference(Parameters params, Parameters features);

double cost_fn(Parameters params, LabeledData dataset);

Parameters get_gradients(Parameters params, LabeledData dataset);

Parameters gradient_descent(LabeledData dataset);

double MAE(Parameters params, LabeledData dataset);

double R_SQUARED(const Parameters &params, const LabeledData &dataset);
