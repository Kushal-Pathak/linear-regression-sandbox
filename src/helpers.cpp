#include "structs.h"
#include <string>
#include <vector>
#include "globals.h"
#include "helpers.h"
#include <iomanip>
#include <iostream>
#include <windows.h>

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

double map_value(double value, double min1, double max1, double min2, double max2)
{
    if (max1 - min1 == 0.0)
    {
        return min2;
    }
    double t = (value - min1) / (max1 - min1);
    return min2 + t * (max2 - min2);
}

void init_graph(std::vector<std::string> &buf, int H, int W)
{
    buf.assign(H, std::string(W, ' '));
    // corners
    buf[0][0] = '+';
    buf[0][W - 1] = '+';
    buf[H - 1][0] = '+';
    buf[H - 1][W - 1] = '+';
    // horizontal
    for (int j = 1; j < W - 1; ++j)
    {
        buf[0][j] = '-';
        buf[H - 1][j] = '-';
    }
    // vertical
    for (int i = 1; i < H - 1; ++i)
    {
        buf[i][0] = '|';
        buf[i][W - 1] = '|';
    }
}
void draw_graph(std::vector<std::string> &buf, int H, int W)
{
    std::string frame;
    frame.reserve(H * (W + 1));

    for (int i = H - 1; i > -1; --i)
    {
        frame.append(buf[i]);
        frame.push_back('\n');
    }

    // Move cursor to top-left and print in one burst
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    COORD home{0, 0};
    SetConsoleCursorPosition(hOut, home);

    std::cout << frame;
    std::cout.flush();
}
