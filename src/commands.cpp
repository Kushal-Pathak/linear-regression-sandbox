#include <iostream>
#include "commands.h"
#include <vector>
#include <string>
#include <sstream>
#include "controllers.h"

std::vector<std::string> split_words(const std::string &sentence)
{
    std::vector<std::string> words;
    std::string word;
    std::istringstream iss(sentence);
    while (iss >> word)
    {
        words.push_back(word);
    }
    return words;
}

void command_processor(std::vector<std::string> commands)
{
    std::string command;
    if (commands.size() == 1)
    {
        command = commands[0];
        if (command == "help")
        {
            help_program();
        }
        else if (command == "about")
        {
            about_program();
        }
        else if (command == "version")
        {
            version_program();
        }
        else if (command == "clear")
        {
            clear_console();
        }
        else if (command == "quit")
        {
            quit_program();
        }
        else
        {
            invalid_command();
        }
    }
    else if (commands.size() > 1)
    {
        command = commands[0];
        if (command == "input")
        {
            feed_input(commands);
        }
        else if (commands.size() == 2)
        {
            command = commands[0] + " " + commands[1];
            if (command == "show models")
            {
                show_models();
            }
            else if (command == "train model")
            {
                train_model();
            }
            else if (command == "save model")
            {
                save_current_model();
            }
            else if (command == "show trash")
            {
                show_trash();
            }
            else if (command == "current model")
            {
                show_current_model();
            }
            else if (command == "eval model")
            {
                evaluate_model();
            }
            else if (command == "show data")
            {
                show_data();
            }
            else if (command == "split data")
            {
                split_data("0.8");
            }
            else
            {
                invalid_command();
            }
        }
        else if (commands.size() > 2)
        {
            command = commands[0] + " " + commands[1];
            if (command == "gen data" && commands.size() == 7)
            {
                gen_data(commands);
            }
            else if (commands.size() == 3)
            {
                command = commands[0] + " " + commands[1];
                if (command == "use model")
                {
                    std::string model_id = commands[2];
                    use_model(model_id);
                }
                else if (command == "new model")
                {
                    std::string model_name = commands[2];
                    new_model(model_name);
                }
                else if (command == "clone model")
                {
                    std::string model_id = commands[2];
                    clone_model(model_id);
                }
                else if (command == "load data")
                {
                    std::string filename = commands[2];
                    load_data(filename);
                }
                else if (command == "delete model")
                {
                    std::string model_id = commands[2];
                    delete_model(model_id);
                }
                else if (command == "restore model")
                {
                    std::string model_id = commands[2];
                    restore_model(model_id);
                }
                else if (command == "save data")
                {
                    std::string filename = commands[2];
                    save_data(filename);
                }
                else if (command == "eval model")
                {
                    std::string model_id = commands[2];
                    evaluate_model(model_id);
                }
                else if (command == "split data")
                {
                    std::string split_ratio = commands[2];
                    split_data(split_ratio);
                }
                else if (command == "show data")
                {
                    std::string arg = commands[2];
                    show_data(arg);
                }
                else if (command == "about model")
                {
                    std::string model_id = commands[2];
                    about_model(model_id);
                }
                else
                {
                    invalid_command();
                }
            }
            else if (commands.size() == 4)
            {
                command = commands[0] + " " + commands[1];

                if (command == "rename model")
                {
                    std::string model_id = commands[2], new_name = commands[3];
                    rename_model(model_id, new_name);
                }
                else
                {
                    invalid_command();
                }
            }
            else
            {
                invalid_command();
            }
        }
    }
}
