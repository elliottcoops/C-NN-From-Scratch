#ifndef METRICS_H
#define METRICS_H

#include "nn_config.h"
#include "data.h"

float find_accuracy(int* predictions, int* actual_values);
void calculate_confusion_matrix(int* actual, int* predicted, int size, int confusion_matrix[][10]);
int write_metrics_to_output(float* loss_per_epoch, int num_elements, float accuracy, double cpu_time, int confusion_matrix[][10], int epochs, float learning_rate);

#endif