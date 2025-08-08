#include "structs.h"
#include <string>
#include <vector>
#include "globals.h"
#include "helpers.h"
#include <iomanip>
#include <iostream>

bool models_are_equal(Model model_a, Model model_b)
{
    if (model_a.id != model_b.id)
    {
        return false;
    }

    if (model_a.name != model_b.name)
    {
        return false;
    }

    if (model_a.parameters.size() != model_b.parameters.size())
    {
        return false;
    }

    int size = model_a.parameters.size();

    for (int i = 0; i < size; i++)
    {
        if (model_a.parameters[i] != model_b.parameters[i])
        {
            return false;
        }
    }

    return true;
}

int find_model_index_by_id(int id, Models models)
{
    for (int i = 0; i < models.size(); i++)
    {
        if (models[i].id == id)
        {
            return i;
        }
    }
    return -1;
}

