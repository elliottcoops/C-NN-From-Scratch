#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "../headers/nn_config.h"
#include "../headers/nn_run.h"
#include "../headers/data.h"

/**
 * Train the neural network passed into the function based on hyper-parameters {learning_rate, epochs}
 * Go for 'epochs' epochs and go through each training example
 */
void train(Layer* neural_network, int layers, float learning_rate, int epochs, float* loss_per_epoch){
    float mse;

    for (int epoch = 0; epoch < epochs; epoch++){
        mse = 0.0;
        for (int example = 0; example < get_train_size(); example++){
            set_input_layer(neural_network, example); // Set the input layer with next training example
            forward_pass(neural_network, layers); // Do forward pass
            mse += back_pass(neural_network, layers, example, learning_rate); // Do backwards pass
        }
        mse /= get_train_size();
        loss_per_epoch[epoch] = mse;

        // Output mse every 100 epochs
        if (epoch % 100 == 0){
            printf("Epoch: %d \t Loss: %.5f\n", epoch, mse);
        }
    }
}

/**
 * Set the input layer for the particular training example passed in
 */
void set_input_layer(Layer* neural_network, int example){
    Neuron** input_layer = neural_network[0].neurons;
    for (int neuron = 0; neuron < get_num_inputs(); neuron++){
        input_layer[neuron]->a = get_train_X_value(neuron, example);
    }
}

/**
 * Carry out forward pass of neural network
 */
void forward_pass(Layer* neural_network, int layers){
    Neuron** current_layer_neurons;
    Neuron** next_layer_neurons;
    Neuron* neuron;
    int num_current_neurons, num_next_neurons;

    for (int i = 0; i < layers - 1; i++) {
        current_layer_neurons = neural_network[i].neurons;
        next_layer_neurons = neural_network[i + 1].neurons;
        num_current_neurons = neural_network[i].num_neurons;
        num_next_neurons = neural_network[i + 1].num_neurons;

        for (int j = 0; j < num_next_neurons; j++) {
            float sum = 0.0;
            for (int k = 0; k < num_current_neurons; k++) {
                neuron = current_layer_neurons[k];
                sum += neuron->a * neuron->weights[j]->w; // Weight the neuron activation with weight to the next neuron
            }
            next_layer_neurons[j]->z = sum; // Set the weighted sum
            next_layer_neurons[j]->a = sigmoid(sum); // Compute activation through sigmoid
        }
    }
}

/**
 * Carry out backwards pass through the network
 */
float back_pass(Layer* neural_network, int layers, int example, float learning_rate){
    int output_layer = layers - 1;
    int num_current_neurons, num_next_neurons;
    Neuron** output_neurons = neural_network[output_layer].neurons;
    Neuron** current_neurons;
    Neuron** next_neurons;
    Neuron* neuron;
    float error, error_sum, mse;
    mse = 0.0;

    // Calculate error and delta for the output layer
    for (int i = 0; i < neural_network[output_layer].num_neurons; i++) {
        neuron = output_neurons[i];
        error = get_train_Y_value(i, example) - neuron->a;
        mse += pow(error, 2); // Add to mse
        neuron->delta = error * sigmoid_derivative(neuron->a);
    }

    // Backpropagate the error and update weights
    for (int i = layers - 2; i >= 0; i--) {
        current_neurons = neural_network[i].neurons;
        next_neurons = neural_network[i + 1].neurons;
        num_current_neurons = neural_network[i].num_neurons;
        num_next_neurons = neural_network[i + 1].num_neurons;

        // Calculate delta for current layer
        for (int j = 0; j < num_current_neurons; j++) {
            neuron = current_neurons[j];
            error_sum = 0.0;
            for (int k = 0; k < num_next_neurons; k++) {
                error_sum += next_neurons[k]->delta * neuron->weights[k]->w;
            }
            neuron->delta = error_sum * sigmoid_derivative(neuron->a);
        }

        // Update weights
        for (int j = 0; j < num_current_neurons; j++) {
            neuron = current_neurons[j];
            for (int k = 0; k < num_next_neurons; k++) {
                neuron->weights[k]->w += learning_rate * neuron->a * next_neurons[k]->delta;
            }
        }
    }

    return mse;
}

/**
 * Get predictions from the predefined test set
 */
void predict(Layer* neural_network, int layers, int* predictions){
    Neuron** input_layer = neural_network[0].neurons;
    int prediction;

    for (int example = 0; example < get_test_size(); example++){
        for (int neuron = 0; neuron < get_num_inputs(); neuron++){
            input_layer[neuron]->a = get_test_X_value(neuron, example);
        }
        forward_pass(neural_network, layers); // Do forward pass
        prediction = get_prediction(neural_network, layers); // Get prediction
        predictions[example] = prediction; // Append to predictions array
    }
}

/**
 * Get the prediction from the neural network output layer
 * Finds the highest value 
 */
int get_prediction(Layer* neural_network, int layers){
    Neuron** output_layer = neural_network[layers-1].neurons;
    int current_highest = 0;
    for (int neuron = 1; neuron < get_num_outputs(); neuron++){
        if (output_layer[neuron]->a > output_layer[current_highest]->a){
            current_highest = neuron;
        }
    }
    return current_highest;
}

/**
 * Sigmoid function
 */
float sigmoid(float value){
    return 1 / (1 + exp(-value));
}

/**
 * Derivative of sigmoid
 */
float sigmoid_derivative(float value){
    return value * (1 - value);
}
