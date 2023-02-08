#include "micrograd.hpp"
#include "Neuron.h"
#include "stdafx.h"

using namespace micrograd;

int main()
{
	std::vector< Value<double> > inputs{ -1.0, 0.2, -0.4, 0.9, 0.1 };

	Neuron neuron(5);

	Value<double> act = neuron(inputs);
	act.back_prop();

	std::cout << act.to_string() << std::endl << std::endl;

	double learning_rate = 0.1;

	for (Value<double>*& param : act.parameters())
		if (param->grad > 0.0)
			param->data += learning_rate;
		else
			param->data -= learning_rate;

	act = neuron(inputs);
	act.zero_grad();
	act.back_prop();

	std::cout << act.to_string() << std::endl << std::endl;

	return 0;
}