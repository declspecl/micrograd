#include "stdafx.h"

#include "Neuron.h"

Neuron::Neuron(unsigned numOfInputs) noexcept
	: bias(0.0)
	, final_sum(0.0)
{
	std::uniform_real_distribution<double> dist(-1.0, 1.0);
	std::default_random_engine gen((unsigned)::time(0));

	this->weights.reserve(numOfInputs);

	for (unsigned i = 0; i < numOfInputs; i++)
		this->weights.push_back(dist(gen));

	this->bias.data = dist(gen);
}

std::vector< micrograd::Value<double>* > Neuron::parameters() noexcept
{
	std::vector< micrograd::Value<double>* > parameters{ &this->bias };
	parameters.reserve(this->weights.size());

	for (micrograd::Value<double>& weight : this->weights)
		parameters.push_back(&weight);
	
	return parameters;
}

std::string Neuron::to_string() const noexcept
{
	std::string weightString = "";

	for (size_t i = 0; i < this->weights.size(); i++)
		if (i < this->weights.size() - 1)
			weightString += this->weights[i].to_string() + ", ";
		else
			weightString += this->weights[i].to_string();

	return std::format("weights: {{{}}}\nbias: {}", weightString, this->bias.to_string());
}

micrograd::Value<double> Neuron::activate(std::vector<micrograd::Value<double>>& inputs) noexcept
{
	this->inputs = inputs;

	this->products.clear();
	this->products.reserve(this->inputs.size() * 1.5);

	this->intermediate_sums.clear();
	this->intermediate_sums.reserve(this->inputs.size());

	for (size_t i = 0; i < inputs.size(); i++)
		this->products.push_back(this->inputs[i] * this->weights[i]);

	if (this->products.size() % 2 != 0)
		this->products.push_back(micrograd::Value<double>(0.0));

	for (size_t i = 0; i + 1 < this->products.size(); i += 2)
		this->intermediate_sums.push_back(this->products[i] + this->products[i + 1]);

	const size_t before_size = intermediate_sums.size();

	for (size_t i = 0; i + 1 < before_size; i += 2)
		this->intermediate_sums.push_back(this->intermediate_sums[i] + this->intermediate_sums[i + 1]);

	this->final_sum = this->intermediate_sums[this->intermediate_sums.size() - 1] + this->bias;

	return this->final_sum.tanh();
}

micrograd::Value<double> Neuron::operator()(std::vector<micrograd::Value<double>>& inputs) noexcept
{
	return this->activate(inputs);
}
