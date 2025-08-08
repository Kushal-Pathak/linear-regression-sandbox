#include "linear_algebra.h"
#include <stdexcept>

double dot_product(const std::vector<double> &a, const std::vector<double> &b)
{
    if (a.size() != b.size())
        throw std::runtime_error("Vector size must match for dot product!");
    double sum = 0;
    for (size_t i = 0; i < a.size(); ++i)
        sum += a[i] * b[i];
    return sum;
}

std::vector<double> add_vectors(const std::vector<double> &a, const std::vector<double> &b)
{
    if (a.size() != b.size())
        throw std::runtime_error("Vector size must match for adding two vectors!");
    std::vector<double> result;
    for (int i = 0; i < a.size(); i++)
    {
        result.push_back(a[i] + b[i]);
    }
    return result;
}

std::vector<double> subtract_vectors(const std::vector<double> &a, const std::vector<double> &b)
{
    if (a.size() != b.size())
        throw std::runtime_error("Vector size must match for subtracting two vectors!");
    std::vector<double> result;
    for (int i = 0; i < a.size(); i++)
    {
        result.push_back(a[i] - b[i]);
    }
    return result;
}

std::vector<double> scale_vector(const std::vector<double> &a, double k)
{
    std::vector<double> result;
    for (int i = 0; i < a.size(); i++)
    {
        result.push_back(k * a[i]);
    }
    return result;
}
