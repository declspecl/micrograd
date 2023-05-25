#pragma once

#include "Neuron.h"

#include <vector>

class Layer
{
public:
	std::vector<Neuron> neurons;

	Layer(unsigned numOfInputs, unsigned numOfOutputs) noexcept;

	std::vector< micrograd::Value<double>* > parameters();

	std::vector< micrograd::Value<double> > activate(std::vector< micrograd::Value<double> >& inputs) noexcept;
	std::vector< micrograd::Value<double> > operator()(std::vector< micrograd::Value<double> >& inputs) noexcept;
};