#include "stdafx.h"

#include "MLP.h"

#include <iostream>

using namespace micrograd;

int main()
{
	std::vector< Value<double> > inputs{ 1.0, 2.0, 3.0 };

	// declaring a MLP with an input layer of length 3, a hidden layer of length 1, and an output layer of length 1
	MLP mlp({ 3, 1, 1 });

	// passing each input through the entire MLP
	std::vector< Value<double> > activations = mlp.activate(inputs);

	
	std::cout << "Output Value: " << std::endl;
	for (const Value<double>& value : activations)
	{
		std::cout << value.to_string() << std::endl;
	}

	return 0;
}