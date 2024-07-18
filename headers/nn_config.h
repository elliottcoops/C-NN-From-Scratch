#ifndef NN_CONFIG_H
#define NN_CONFIG_H

typedef struct Neuron {
    float z;
    float a;
    float delta;
    int num_weights;
    struct Weight** weights;
} Neuron;

typedef struct Weight {
    float w;
} Weight;

typedef struct Layer {
    int num_neurons;
    struct Neuron** neurons;
} Layer;

Layer* initialise_network(int architecture[], int layers);
void load_file_data(const char filename[], float max_value, int max_row_length, int rows, int cols, float arr[rows][cols]);
Neuron** initialise_neurons(int number_of_neurons, int next_number_of_neurons);
Weight** initialise_weights(int number_of_weights);
float randn();
void debug_network(Layer* network, int layers);

#endif 
