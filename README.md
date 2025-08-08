# Linear Regression Sandbox

This is a C++ terminal-based linear regression sandbox made for learning and trying ideas. You can make synthetic data or load real-world data, train the model with gradient descent, adjust the learning rate, and make predictions — all using simple console commands. The executable runs on Windows only. The code compiles on Windows, but requires minor modifications to work on other platforms. All datasets should be saved and loaded from the datasets folder in CSV format. All models are saved in the models folder inside the stored_models file.

## Steps to compile and run the project

### Compile

```
g++ -Iinclude main.cpp src/random.cpp src/commands.cpp src/controllers.cpp src/linear_algebra.cpp src/linear_regression.cpp src/data_generator.cpp src/print_utils.cpp src/helpers.cpp src/globals.cpp src/csv.cpp -o app
```

or

```
g++ "@build.txt"
```

---

### Run

```
./app
```

##
