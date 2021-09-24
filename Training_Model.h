#pragma once
#include <Eigen/Dense>

class Training_Model
{
private:
	int num_of_samples;
	int* layers;
	int num_of_layers;
	int num_of_features;
	int* num_of_neurons;
	Eigen::MatrixXd* Ws;
	Eigen::MatrixXd* Zs;
	Eigen::VectorXd* bs;
	Eigen::MatrixXd* dWs;
	Eigen::MatrixXd* dZs;
	Eigen::VectorXd* dbs;
	Eigen::MatrixXd* As;
	Eigen::MatrixXd dA;
	Eigen::MatrixXd X;
	Eigen::MatrixXd Y;
public:
	Training_Model(int num_of_features, int num_of_layers, int* num_of_neurons, int num_of_samples, Eigen::MatrixXd X, Eigen::MatrixXd Y);
	~Training_Model();
	void L_forward();
	void L_backward();
	void update(double learning_rate); 
	void train(double learning_rate, int iteration, bool print); 
	double cost();
	void test(Eigen::MatrixXd test_X, Eigen::MatrixXd test_Y, int row, int column); //not implemented
};