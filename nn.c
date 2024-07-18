// main.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "headers/nn_config.h"
#include "headers/nn_run.h"
#include "headers/data.h"
#include "headers/metrics.h"
#include "headers/nn.h"

int main(int argc, char** argv){
    // Hyperparameters
    float learning_rate; 
    int epochs;
    int layers = 2 + (argc - 3);
    int* architecture  = (int*)calloc(layers, sizeof(int));
    read_cmd_line_args(&learning_rate, &epochs, layers, architecture, argc, argv); // Read in variables from cmdl

    // Logging information
    int confusion_matrix[get_num_outputs()][get_num_outputs()];
    float accuracy;
    float loss_per_epoch[epochs];
    clock_t start, end;
    double cpu_time_used;

    // Test/Prediction 
    int* predictions = (int*)calloc(get_test_size(), sizeof(int));
    int* actual_values = (int*)calloc(get_test_size(), sizeof(int));

    // Neural network
    Layer* neural_network = initialise_network(architecture, layers); 

    // Load dataset
    load_dataset(); 

    // Train the neural network with learning rate and epochs and record loss per epoch
    // Record the time taken to train as well
    start = clock(); 
    train(neural_network, layers, learning_rate, epochs, loss_per_epoch); 
    end = clock();
    cpu_time_used = ((double) (end-start)) / CLOCKS_PER_SEC;
    
    // Predict based on the defined train set and get actual values
    predict(neural_network, layers, predictions);
    get_actual_test_output(actual_values);

    // Find predictions
    accuracy = find_accuracy(predictions, actual_values);
    calculate_confusion_matrix(actual_values, predictions, get_test_size(), confusion_matrix);

    // Write metrics to output
    write_metrics_to_output(loss_per_epoch, epochs, accuracy, cpu_time_used, confusion_matrix, epochs, learning_rate);

    // Write image, prediction and actual to output for visualisation
    write_pred_and_test(predictions, actual_values);

    return 0;
}

/**
 * Read in (hyper)parameters from the command line
 */
void read_cmd_line_args(float* lr, int* epochs, int layers, int* architecture, int num_args, char** args){
    // Set learning rate and epochs
    *lr = atof(args[1]);
    *epochs = atoi(args[2]);

    // Set architecture
    architecture[0] = get_num_inputs();
    architecture[layers-1] = get_num_outputs();
    for (int i = 3; i < num_args; i++){
        architecture[i-2] = atoi(args[i]);
    }
}

