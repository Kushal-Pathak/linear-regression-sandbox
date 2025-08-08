#include "random.h"
#include <random>

double generate_random_double(double min, double max)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(min, max);
    return dist(gen);
}

std::vector<double> generate_one_random_vector(int num_features, double min, double max)
{
    std::vector<double> v;
    v.reserve(num_features);
    for (int i = 0; i < num_features; ++i)
        v.push_back(generate_random_double(min, max));
    return v;
}
