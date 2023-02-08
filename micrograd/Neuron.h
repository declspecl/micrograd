#pragma once

#include "micrograd.hpp"

#include <vector>
#include <chrono>
#include <string>
#include <format>
#include <numeric>
#include <algorithm>

class Neuron
{
private:
	std::vector< micrograd::Value<double> > inputs;
	std::vector< micrograd::Value<double> > products;
	std::vector< micrograd::Value<double> > intermediate_sums;
	micrograd::Value<double> final_sum;

public:
	std::vector< micrograd::Value<double> > weights;
	micrograd::Value<double> bias;

	Neuron(unsigned numOfInputs) noexcept;

	std::vector< micrograd::Value<double>* > parameters() noexcept;

	std::string to_string() const noexcept;

	micrograd::Value<double> activate(std::vector< micrograd::Value<double> >& inputs) noexcept;
	micrograd::Value<double> operator()(std::vector< micrograd::Value<double> >& inputs) noexcept;
};