#include "micrograd.hpp"
#include "Neuron.h"

#include <stdio.h>

using namespace micrograd;

int main()
{
	Neuron neuron(3);

	std::vector< Value<double> > inputs{ 1.0, 2.0, 3.0 };

	neuron.weights = { 3.0, 2.0, 1.0 };

	Value<double> activation = neuron.activate(inputs);

	std::cout << "neuron bias: " << std::endl;
	std::cout << neuron.bias.toString() << std::endl << std::endl;

	std::cout << "neuron weights: " << std::endl;
	for (int i = 0; i < neuron.weights.size(); i++)
		std::cout << neuron.weights[i].toString() << std::endl;
	
	std::cout << std::endl << std::endl;

	activation.backPropagate();

	for (const auto& child : activation.parameters())
	{
		std::cout << child->toString() << std::endl << std::endl;
	}

	return 0;
}