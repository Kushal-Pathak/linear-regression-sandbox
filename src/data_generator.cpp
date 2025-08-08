#include <iostream>
#include "data_generator.h"
#include "random.h"
#include "linear_algebra.h"
#include "print_utils.h"
#include "linear_regression.h"
#include "globals.h"

LabeledData generate_dataset(int num_samples, int num_features, double feature_min, double feature_max, double noiseLevel)
{
    LabeledData dataset;

    // generate random model parameters
    GENERATED_PARAMETERS = generate_one_random_vector(num_features + 1, -10, 10);

    for (int i = 0; i < num_samples; ++i)
    {
        // Generate one random features vector
        std::vector<double> features = generate_one_random_vector(num_features, feature_min, feature_max);

        // Calculate output
        double y = inference(GENERATED_PARAMETERS, features);

        // Add noise term to the output
        double noise_term = generate_random_double(-noiseLevel, noiseLevel);
        y = y + y * noise_term;

        dataset.x.push_back(std::move(features));
        dataset.y.push_back(y);
    }

    return dataset;
}
