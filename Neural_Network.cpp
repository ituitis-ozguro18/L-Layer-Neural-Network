// Neural_Network.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include "Training_Model.h"


#define TRAINING_EXAMPLES 250
#define TEST_EXAMPLES 47
#define NUM_LAYERS 4

using namespace std;

void split(vector<string>& to_push, string to_split);
void split_row(vector<double>& to_push, string to_split);

int main()
{
    vector<string> features;
    string column;
    ifstream datafeed("heart.csv");
    getline(datafeed, column);
    cout << column << endl;;

    split(features, column);

    int num_of_features = features.size() - 1;
    Eigen::MatrixXd X_Training(num_of_features, TRAINING_EXAMPLES);
    Eigen::MatrixXd Y_Training(1, TRAINING_EXAMPLES); 
    Eigen::MatrixXd X_Test(num_of_features, TEST_EXAMPLES);
    Eigen::MatrixXd Y_Test(1, TEST_EXAMPLES);
  
    string row;
    vector<double> temp_to_convert;
    for (int i = 0; i < TRAINING_EXAMPLES; i++)
    {
        row = "";
        temp_to_convert.clear();
        getline(datafeed, row);
        split_row(temp_to_convert, row);
       
        for (int j = 0; j < temp_to_convert.size()-1; j++)
        {
            X_Training(j,i) = temp_to_convert[j];
        }
        Y_Training(0, i) = temp_to_convert[temp_to_convert.size() - 1];
    }
    

    for (int i = 0; i < TEST_EXAMPLES; i++)
    {
        row = "";
        temp_to_convert.clear();
        getline(datafeed, row);       
        split_row(temp_to_convert, row);

        for (int j = 0; j < temp_to_convert.size() - 1; j++)
        {
            X_Test(j, i) = temp_to_convert[j];
        }
        Y_Test(0, i) = temp_to_convert[temp_to_convert.size() - 1];      
    }

    //Set-up the number of neurons for each layer
    int* num_of_neurons = new int[NUM_LAYERS];
    num_of_neurons[0] = 8;
    num_of_neurons[1] = 4;
    num_of_neurons[2] = 2;
    num_of_neurons[3] = 1;


    //Turn the set into a binary classification problem
    for (int i = 0; i < TRAINING_EXAMPLES; i++)
    {
        if (Y_Training(0, i) >= 2) Y_Training(0, i) = 1;
        else Y_Training(0, i) = 0;
    }

    //Train and test with different parameters
    Training_Model model1(num_of_features, NUM_LAYERS, num_of_neurons, TRAINING_EXAMPLES, X_Training, Y_Training);
    model1.train(0.05, 100, true);
    model1.test(X_Test, Y_Test, num_of_neurons[NUM_LAYERS - 1], TEST_EXAMPLES);

    Training_Model model2(num_of_features, NUM_LAYERS, num_of_neurons, TRAINING_EXAMPLES, X_Training, Y_Training);
    model2.train(10, 40, true);
    model2.test(X_Test, Y_Test, num_of_neurons[NUM_LAYERS - 1], TEST_EXAMPLES);

    Training_Model model3(num_of_features, NUM_LAYERS, num_of_neurons, TRAINING_EXAMPLES, X_Training, Y_Training);
    model3.train(0.0001, 100, true);
    model3.test(X_Test, Y_Test, num_of_neurons[NUM_LAYERS - 1], TEST_EXAMPLES);

}


// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
