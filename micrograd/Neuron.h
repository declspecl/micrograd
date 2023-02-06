#pragma once

#include "micrograd.hpp"

#include <vector>
#include <random>
#include <chrono>

class Neuron
{
public:
	std::vector< micrograd::Value<double> > weights;
	micrograd::Value<double> bias;

	Neuron(unsigned numberOfInputs) noexcept;
	
	std::vector< micrograd::Value<double>* > parameters() noexcept;
	micrograd::Value<double> activate(std::vector< micrograd::Value<double> > inputs) noexcept;
};