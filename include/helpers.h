#include "structs.h"

bool models_are_equal(Model model_a, Model model_b);

int find_model_index_by_id(int id, Models models = SAVED_MODELS);

double map_value(double value, double min1, double max1, double min2, double max2);

void init_graph(std::vector<std::string> &buf, int H, int W);

void draw_graph(std::vector<std::string> &buf, int H, int W);
