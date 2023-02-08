#pragma once

#include "Neuron.h"

#include <vector>

class Layer
{
private:
	std::vector< micrograd::Value<double> > activations;
	micrograd::Value<double> final_sum;

public:
	std::vector<Neuron> neurons;

	Layer(unsigned numOfInputs, unsigned numOfOutputs) noexcept;

	micrograd::Value<double> activate(std::vector< micrograd::Value<double> >& inputs) noexcept;
	micrograd::Value<double> operator()(std::vector< micrograd::Value<double> >& inputs) noexcept;
};