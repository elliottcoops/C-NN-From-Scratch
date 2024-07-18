#ifndef DATA_H
#define DATA_H

void load_dataset();
void load_test_and_train();
float get_dataset_X_value(int i, int j);
float get_dataset_Y_value(int i, int j);
float get_test_X_value(int i, int j);
float get_test_Y_value(int i, int j);
float get_train_X_value(int i, int j);
float get_train_Y_value(int i, int j);
int get_num_inputs();
int get_num_outputs();
int get_train_size();
int get_dataset_size();
int get_test_size();
void get_actual_test_output(int* actual_values);
void load_file_data(const char filename[], float max_value, int max_row_length, int rows, int cols, float arr[rows][cols]);

#endif