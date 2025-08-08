#include <iostream>
#include <cstdlib>
#include "commands.h"
#include "controllers.h"

int main()
{
    init();
    system("cls");
    std::cout << "Welcome to Linear Regression Sandbox\n";
    std::string sentence;
    while (1)
    {
        std::cout << "LR ~ ";
        std::getline(std::cin, sentence);
        auto commands =  split_words(sentence);
        command_processor(commands);
    }

    return 0;
}