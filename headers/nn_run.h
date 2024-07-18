#ifndef NN_RUN_H
#define NN_RUN_H

void train(Layer* neural_network, int layers, float learning_rate, int epochs, float* loss_per_epoch);
void set_input_layer(Layer* neural_network, int example);
void forward_pass(Layer* neural_network, int layers);
float back_pass(Layer* neural_network, int layers, int example, float learning_rate);
void predict(Layer* neural_network, int layers, int* predictions);
int get_prediction(Layer* neural_network, int layers);
float sigmoid(float value);
float sigmoid_derivative(float value);

#endif