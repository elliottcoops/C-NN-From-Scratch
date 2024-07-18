// main.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "headers/nn_config.h"
#include "headers/nn_run.h"
#include "headers/data.h"
#include "headers/metrics.h"

// Hyperparameters
const int epochs = 1000;
const float learning_rate = 0.1; 

int main(int argc, char** argv){
    int architecture[] = {get_num_inputs(), 32, 16, get_num_outputs()}; // Set architecture
    int confusion_matrix[get_num_outputs()][get_num_outputs()];
    int* predictions = (int*)calloc(get_test_size(), sizeof(int));
    int* actual_values = (int*)calloc(get_test_size(), sizeof(int));
    int layers = sizeof(architecture)  / sizeof(int); // Find how many layers
    Layer* neural_network = initialise_network(architecture, layers); // Initialise the neural network
    float accuracy;
    float loss_per_epoch[epochs];
    clock_t start, end;
    double cpu_time_used;

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

    return 0;
}