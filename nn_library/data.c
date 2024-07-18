#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "../headers/data.h"

const char INPUT_FILE[] = "data/mnist-input.csv";
const char OUTPUT_FILE[] = "data/mnist-output.csv";

const int NUM_INPUTS = 64;
const int NUM_OUTPUTS = 10;
const int NUM_TEST_EXAMPLES = 297;
const int NUM_TRAINING_EXAMPLES = 1500;
const int NUM_DATASET_EXAMPLES = 1797;

const float INPUT_FILE_MAX_VALUE = 16.0; 
const int INPUT_FILE_MAX_ROW_LENGTH = 10000;

const float OUTPUT_FILE_MAX_VALUE = 1.0; 
const int OUTPUT_FILE_MAX_ROW_LENGTH = 10000;

const char DELIMITER[] = ",";

float dataset_X[NUM_INPUTS][NUM_DATASET_EXAMPLES]; 
float dataset_Y[NUM_OUTPUTS][NUM_DATASET_EXAMPLES]; 
float test_X[NUM_INPUTS][NUM_TEST_EXAMPLES]; 
float test_Y[NUM_OUTPUTS][NUM_TEST_EXAMPLES]; 
float train_X[NUM_INPUTS][NUM_TRAINING_EXAMPLES]; 
float train_Y[NUM_OUTPUTS][NUM_TRAINING_EXAMPLES]; 

/**
 * Load data into datasets
 */
void load_dataset(){
    load_file_data(INPUT_FILE, INPUT_FILE_MAX_VALUE, INPUT_FILE_MAX_ROW_LENGTH, NUM_INPUTS, NUM_DATASET_EXAMPLES, dataset_X);
    load_file_data(OUTPUT_FILE, OUTPUT_FILE_MAX_VALUE, OUTPUT_FILE_MAX_ROW_LENGTH, NUM_OUTPUTS, NUM_DATASET_EXAMPLES, dataset_Y);
    load_test_and_train();
}

/** Load data into test and training sets */
void load_test_and_train(){
    // Load into train sets
    for (int i = 0; i < NUM_TRAINING_EXAMPLES; i++){
        for (int j = 0; j < NUM_INPUTS; j++){
            train_X[j][i] = get_dataset_X_value(j,i);
        }

        for (int j = 0; j < NUM_OUTPUTS; j++){
            train_Y[j][i] = get_dataset_Y_value(j,i);
        }
    }

    // Load into test sets
    for (int i = NUM_TRAINING_EXAMPLES; i < NUM_DATASET_EXAMPLES; i++){
        for (int j = 0; j < NUM_INPUTS; j++){
            test_X[j][i-NUM_TRAINING_EXAMPLES] = get_dataset_X_value(j,i);
        }

        for (int j = 0; j < NUM_OUTPUTS; j++){
            test_Y[j][i-NUM_TRAINING_EXAMPLES] = get_dataset_Y_value(j,i);
        }
    }
}

float get_dataset_X_value(int i, int j){ return dataset_X[i][j];}

float get_dataset_Y_value(int i, int j){ return dataset_Y[i][j];}

float get_test_X_value(int i, int j){ return test_X[i][j];}

float get_test_Y_value(int i, int j){ return test_Y[i][j];}

float get_train_X_value(int i, int j){ return train_X[i][j];}

float get_train_Y_value(int i, int j){ return train_Y[i][j];}

int get_num_inputs(){ return NUM_INPUTS;}

int get_num_outputs(){ return NUM_OUTPUTS;}

int get_train_size(){ return NUM_TRAINING_EXAMPLES;}

int get_dataset_size(){return NUM_DATASET_EXAMPLES;}

int get_test_size(){ return NUM_TEST_EXAMPLES;}

/**
 * Get the expected digits the test set
 */
void get_actual_test_output(int* actual_values){
    for (int example = 0; example < get_test_size(); example++){
        for (int digit = 0; digit < get_num_outputs(); digit++){
            if (get_test_Y_value(digit, example) == 1){
                actual_values[example] = digit;
            }
        }
    }
}

/**
 * Write 12 images, actual and predicted labels to output file
 */
int write_pred_and_test(int* predictions, int* actual_values){
    FILE *file = fopen("test_output.txt", "w");
    if (file == NULL) {
        perror("Error opening file");
        return 1;
    }

    // Write first 12 images with actual and predicted labels for displaying
    for (int example = 0; example < 144; example+=12){
        fprintf(file, "---test_example_%d---\n", example+1);
        // Write image data
        fprintf(file, "image: [");
        for (int pixel = 0; pixel < get_num_inputs(); pixel++){
            if (pixel  == get_num_inputs()-1){ fprintf(file, "%f]\n", test_X[pixel][example]);}
            else{ fprintf(file, "%f, ", test_X[pixel][example]);}
        }
        // Write actual value
        fprintf(file, "actual: %d\n", actual_values[example]);
        // Write predicted value
        fprintf(file, "predicted: %d\n", predictions[example]);
        fprintf(file, "---------------------\n\n");
    }

    return 0;
}
/**
 * Load the data from the csv file into the dataset
 * Data is columns of vectors with the data
 */
void load_file_data(const char filename[], float max_value, int max_row_length, int rows, int cols, float arr[rows][cols]){
    FILE *file_ptr = fopen(filename, "r");
    char row_buffer[max_row_length];
    char *token;
    int row, col;

    row = col = 0;

    while (fscanf(file_ptr, "%s", row_buffer) == 1){    
        token = strtok(row_buffer, DELIMITER);
        while (token != NULL) {
            arr[row][col++] = atof(token) / max_value;
            token = strtok(NULL, DELIMITER);
        }
        row++;
        col = 0;
    }
}