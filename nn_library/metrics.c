#include <stdio.h>
#include <stdlib.h>
#include "../headers/metrics.h"
#include "../headers/nn_config.h"
#include "../headers/data.h"
#include "../headers/nn_run.h"

/**
 * Find the accuracy of the **TRAINED** neural network
 * Return accuracy as a fraction in [0,1]
 */
float find_accuracy(int* predictions, int* actual_values){
    int predicted_output, actual_output;
    float accuracy = 0.0;

    for (int example = 0; example < get_test_size(); example++){
        predicted_output = predictions[example];
        actual_output = actual_values[example];
        if (predicted_output == actual_output){ accuracy += 1;} // Add one to accuracy if the predicted and actual value match
    }

    return accuracy / get_test_size();
}

/**
 * Calculate a confusion matrix based on the actual and predicted results
 */
void calculate_confusion_matrix(int* actual, int* predicted, int size, int confusion_matrix[get_num_outputs()][get_num_outputs()]){
    // Initialize confusion matrix to zero
    for (int i = 0; i < get_num_outputs(); ++i) {
        for (int j = 0; j < get_num_outputs(); ++j) {
            confusion_matrix[i][j] = 0;
        }
    }
    
    // Fill confusion matrix
    for (int i = 0; i < size; ++i) {
        int true_class = actual[i];
        int predicted_class = predicted[i];
        
        confusion_matrix[true_class][predicted_class]++;
    }
}

/**
 * Write all metrics calculated to output file for further analysis
 */
int write_metrics_to_output(float* loss_per_epoch, int num_elements, float accuracy, double cpu_time, int confusion_matrix[get_num_outputs()][get_num_outputs()], int epochs, float learning_rate){
    FILE *file = fopen("training_statistics.txt", "w");
    if (file == NULL) {
        perror("Error opening file");
        return 1;
    }

    // Neural network metrics
    fprintf(file, "---POST_TRAINING_STATISTICS---");
    fprintf(file, "\naccuracy: %.2f%%\n", accuracy*100); // Write accuracy
     // Write loss
    fprintf(file, "loss_per_epoch = [");
    for (int i = 0; i < num_elements; i++) {
        if (i==num_elements-1){
            fprintf(file, "%f]\n", loss_per_epoch[i]);
        } else{
            fprintf(file, "%f,", loss_per_epoch[i]);
            }
    }

    // Write confusion matrix to file
    fprintf(file, "confusion_matrix:\n");
    fprintf(file, "\t"); // Print column headers
    for (int c = 0; c < get_num_outputs(); ++c) {
        fprintf(file, "%d\t", c);
    }
    fprintf(file, "\n");
    
    for (int i = 0; i < get_num_outputs(); ++i) {
        fprintf(file, "%d\t", i); // Print row header
        
        for (int j = 0; j < get_num_outputs(); ++j) {
            fprintf(file, "%d\t", confusion_matrix[i][j]);
        }
        
        fprintf(file, "\n");
    }
    fprintf(file, "------------------------------\n");

    fprintf(file, "\n---TRAINING_DETAILS---");
    fprintf(file, "\ncpu_time_used: %.2f\n", cpu_time);  // Write cpu time use
    fprintf(file, "epochs: %d\n", epochs); // Write epochs
    fprintf(file, "learning_rate: %.2f\n", learning_rate); // Write learning rate 
    fprintf(file, "-----------------------\n");

    fclose(file);

    return 0;
}