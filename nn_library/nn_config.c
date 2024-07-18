#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "../headers/nn_config.h"

/**
 * Initialise the neural network with the architecture passed in
 */
Layer* initialise_network(int architecture[], int layers){
    Layer* neural_network = (Layer*)calloc(layers, sizeof(Layer));
    int number_of_neurons, next_number_of_neurons;
    
    for (int i = 0; i < layers; i++){
        number_of_neurons = architecture[i]; // Get number of neurons in current layer
        next_number_of_neurons = (i == layers-1) ? 0 : architecture[i+1]; // Get number of neurons in next layer, 0 if output layer
        neural_network[i].num_neurons = number_of_neurons;
        neural_network[i].neurons = initialise_neurons(number_of_neurons, next_number_of_neurons);
    }
    
    return neural_network;
}

/**
 * Allocate and set memory and data for neurons
 */
Neuron** initialise_neurons(int number_of_neurons, int next_number_of_neurons){
    Neuron** neurons = (Neuron**)calloc(number_of_neurons, sizeof(Neuron*)); // Allocate memory for neurons
    Neuron* new_neuron;
    
    for (int i = 0; i < number_of_neurons; i++){
        new_neuron = (Neuron*)malloc(sizeof(Neuron));
        new_neuron->z = 0.0;
        new_neuron->a = 0.0;    
        new_neuron->delta = 0.0;
        new_neuron->num_weights = next_number_of_neurons;
        new_neuron->weights = initialise_weights(next_number_of_neurons); // Initialise weights to be connected to next set of neurons
        neurons[i] = new_neuron;
    }

    return neurons;
}

/**
 * Initialise weights for neurons
 */
Weight** initialise_weights(int number_of_weights){
    Weight** weights = (Weight**)calloc(number_of_weights, sizeof(Weight*)); // Allocate memory for weights
    Weight* new_weight;

    for (int i = 0; i < number_of_weights; i++){
        new_weight = (Weight*)malloc(sizeof(Weight)); // Create new weight
        new_weight->w = randn(); // Assign random value to weight
        weights[i] = new_weight;
    }

    return weights;
}

/**
 * Generate random number in normal distrubution
 */ 
float randn(){
    float u1 = rand() / (float)RAND_MAX;
    float u2 = rand() / (float)RAND_MAX;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/**
 * Debug neural network with data
 */
void debug_network(Layer* neural_network, int layers){
    for (int i = 0; i < layers; i++){
        printf("Number of neurons in layer %d = %d\n", i, neural_network[i].num_neurons);
        for (int j = 0; j < neural_network[i].num_neurons; j++){
            Neuron* neuron = neural_network[i].neurons[j];
            printf("----Neuron----\n");
            printf("z: %f\n", neuron->z);
            printf("a: %f\n", neuron->a);
            printf("delta: %f\n", neuron->delta);
            printf("number of weights: %d\n", neuron->num_weights);
            printf("----Weights of neuron----\n");
            Weight** weights = neuron->weights;
            for (int k = 0; k < neuron->num_weights; k++){
                printf("weight value: %f\n", weights[k]->w);
            }
            printf("--------\n");
            printf("--------\n");
            printf("\n");
        }
        printf("\n");
    }
}



