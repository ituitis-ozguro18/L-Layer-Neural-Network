#pragma once
#include "Training_Model.h"
#include <iostream>
#include <Eigen/Dense>
#include <math.h>


using namespace Eigen;
using namespace std;

double sigmoid(double x);
double Tanh(double x);

Training_Model::Training_Model(int num_of_features, int num_of_layers, int* num_of_neurons, int num_of_samples, Eigen::MatrixXd X, Eigen::MatrixXd Y)
{
	Training_Model::num_of_neurons = new int[num_of_layers];
	for (int i = 0; i < num_of_layers; i++)
	{
		Training_Model::num_of_neurons[i] = num_of_neurons[i];
	}

	Training_Model::num_of_layers = num_of_layers;
	Training_Model::num_of_features = num_of_features;
	Training_Model::num_of_samples = num_of_samples;
	Training_Model::X = X;
	Training_Model::Y = Y;

	Ws = new MatrixXd[num_of_layers];
	Zs = new MatrixXd[num_of_layers];
	(Ws[0]) = (MatrixXd::Random(num_of_neurons[0], num_of_features));
	(Zs[0]) = (MatrixXd::Zero(num_of_neurons[0], num_of_samples));
	for (int i = 1; i < num_of_layers; i++)
	{
		(Ws[i]) = MatrixXd::Random(num_of_neurons[i], num_of_neurons[i - 1]);
		(Zs[i]) = MatrixXd::Zero(num_of_neurons[i], num_of_samples);
	}

	bs = new VectorXd[num_of_layers];
	for (int i = 0; i < num_of_layers; i++)
	{
		bs[i] = VectorXd::Zero(num_of_neurons[i]);
	}

	As = new MatrixXd[num_of_layers];
	for (int i = 1; i < num_of_layers; i++)
	{
		(As[i]) = MatrixXd::Zero(num_of_neurons[i], num_of_samples);
	}
	dA = MatrixXd::Zero(1, num_of_samples);

	dWs = new MatrixXd[num_of_layers];
	dZs = new MatrixXd[num_of_layers];
	(dWs[0]) = (MatrixXd::Zero(num_of_neurons[0], num_of_features));
	(dZs[0]) = (MatrixXd::Zero(num_of_neurons[0], num_of_samples));
	for (int i = 1; i < num_of_layers; i++)
	{
		(dWs[i]) = MatrixXd::Zero(num_of_neurons[i], num_of_neurons[i - 1]);
		(dZs[i]) = MatrixXd::Zero(num_of_neurons[i], num_of_samples);
	}

	dbs = new VectorXd[num_of_layers];
	for (int i = 0; i < num_of_layers; i++)
	{
		dbs[i] = VectorXd::Zero(num_of_neurons[i]);
	}

}

Training_Model::~Training_Model()
{
	delete[] layers;
	delete[] num_of_neurons;
	delete[] Ws;
	delete[] Zs;
	delete[] bs;
	delete[] dWs;
	delete[] dZs;
	delete[] dbs;
}

void Training_Model::L_forward()
{
	Zs[0] = Ws[0] * X;
	Zs[0].colwise() += bs[0];
	As[0] = Zs[0].array().tanh();

	for (int i = 1; i < num_of_layers; i++)
	{
		Zs[i] = Ws[i] * As[i - 1];
		Zs[i].colwise() += bs[i];
		As[i] = Zs[i].array().tanh();
	}

	for (int i = 0; i < num_of_neurons[num_of_layers - 1]; i++)
	{
		for (int j = 0; j < num_of_samples; j++)
		{
			As[num_of_layers - 1](i, j) = sigmoid(Zs[num_of_layers - 1](i, j));
		}
	}
}

void Training_Model::L_backward()
{

	for (int i = 0; i < num_of_samples; i++)
	{
		dA(0, i) = -((Y(0, i) / As[num_of_layers - 1](0, i)) - (1 - Y(0, i) / 1 - As[num_of_layers - 1](0, i)));
	}

	dZs[num_of_layers - 1] = As[num_of_layers - 1] - Y;

	dWs[num_of_layers - 1] = (1 / (double)num_of_samples) * (dZs[num_of_layers - 1] * As[num_of_layers - 2].transpose());

	double sum = dZs[num_of_layers - 1].sum();
	sum /= (double)num_of_samples;
	for (int i = 0; i < num_of_neurons[num_of_layers - 1]; i++)
	{
		dbs[num_of_layers - 1][i] = sum;
	}

	for (int i = num_of_layers - 1; i > 1; i--)
	{
		dZs[i - 1] = (Ws[i].transpose() * dZs[i]);
		

		for (int k = 0; k < num_of_neurons[i - 1]; k++)
		{
			for (int j = 0; j < num_of_samples; j++)
			{
				dZs[i-1](k, j) *= (1 - As[i-1](k, j) * As[i - 1](k, j));
			}
		}

		dWs[i - 1] = (1 / (double)num_of_samples) * dZs[i - 1] * (As[i - 2].transpose());

		sum = dZs[i - 1].sum();
		sum /= (double)num_of_samples;
		for (int h = 0; h < num_of_neurons[i - 1]; h++)
		{
			dbs[i - 1][h] = sum;
		}
	}

 	dZs[0] = (Ws[1].transpose() * dZs[1]);

	for (int k = 0; k < num_of_neurons[0]; k++)
	{
		for (int j = 0; j < num_of_samples; j++)
		{
			dZs[0](k, j) *= (1 - As[0](k, j) * As[0](k, j));
		}
	}
	dWs[0] = (1 / (double)num_of_samples) * dZs[0] * (X.transpose());

	sum = dZs[0].sum();
	sum /= (double)num_of_samples;
	for (int h = 0; h < num_of_neurons[0]; h++)
	{
		dbs[0][h] = sum;
	}
}

void Training_Model::update(double learning_rate)
{
	for (int i = num_of_layers - 1; i > -1; i--)
	{
		Ws[i] -= learning_rate * dWs[i];
		bs[i] -= learning_rate * dbs[i];
	}
}

void Training_Model::train(double learning_rate, int iteration, bool print)
{
	for (int i = 0; i < iteration; i++)
	{
		L_forward();
		L_backward();
		update(learning_rate);
		if (print) cout << "For iteration number " << i + 1 << " with learning rate " << learning_rate << " cost is " << cost() << "." << endl;
	}
}

double Training_Model::cost()
{
	double cost = 0;

	for (int i = 0; i < num_of_samples; i++)
	{
	cost += Y(0, i) * log(As[num_of_layers - 1](0, i) + 0.00000000000001) + (1 - Y(0, i)) * log(1.00000000000001 - As[num_of_layers - 1](0, i));

	}

	cost /= -(double)num_of_samples;
	return cost;
}

void Training_Model::test(Eigen::MatrixXd test_X, Eigen::MatrixXd test_Y, int row, int column)
{

	num_of_samples = column;
	Zs[0] = Ws[0] * test_X;
	Zs[0].colwise() += bs[0];
	As[0] = Zs[0].array().tanh();

	for (int i = 1; i < num_of_layers; i++)
	{
		Zs[i] = Ws[i] * As[i - 1];
		Zs[i].colwise() += bs[i];
		As[i] = Zs[i].array().tanh();
	}

	for (int i = 0; i < num_of_neurons[num_of_layers - 1]; i++)
	{
		for (int j = 0; j < num_of_samples; j++)
		{
			As[num_of_layers - 1](i, j) = sigmoid(Zs[num_of_layers - 1](i, j));
			Y(i, j) = As[num_of_layers - 1](i, j);
		}
	}

	int trues = 0;
	int falses = 0;
	for (int j = 0; j < row; j++)
	{
		for (int i = 0; i < column; i++)
		{
			if (Y(j, i) >= 0.5)
			{
				if (test_Y(j, i) == 1) trues++;
				else falses++;
			}
			else
			{
				if (test_Y(j, i) == 0) trues++;
				else falses++;
			}
		}
	}
	

	cout << "True Results:" << trues << endl << "False Results:" << falses << endl;
}
