#include <string>
#include <vector>

void init();
void show_logs();
void invalid_command();

void help_program();
void about_program();
void version_program();
void clear_console();
void quit_program();
void load_data(std::string filename);
void split_data(std::string split_ratio);
void show_data(std::string arg = "");
void new_model(std::string model_name);
void train_model();
void evaluate_model(std::string model_id = "");
void save_current_model();
void save_changes_to_file();
void show_models();
void use_model(std::string model_id);
void show_current_model();
void feed_input(std::vector<std::string> args);
void rename_model(std::string model_id, std::string new_name);
void delete_model(std::string model_id);
void show_trash();
void restore_model(std::string model_id);
void clone_model(std::string model_id);
void gen_data(std::vector<std::string> args);
void save_data(std::string filename);
void about_model(std::string model_id);