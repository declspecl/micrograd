#include "Neuron.h"

Neuron::Neuron(unsigned numberOfInputs) noexcept
	: bias(0.0)
{
	std::uniform_real_distribution dist(-1.0, 1.0);
	std::default_random_engine generator((unsigned)::time(0));

	this->weights.reserve(numberOfInputs);

	for (unsigned i = 0; i < numberOfInputs; i++)
		this->weights.push_back(dist(generator));

	this->bias = dist(generator);
}

std::vector< micrograd::Value<double>* > Neuron::parameters() noexcept
{
	std::vector< micrograd::Value<double>* > params{ &this->bias };

	for (micrograd::Value<double>& weight : this->weights)
		params.push_back(&weight);

	return params;
}

micrograd::Value<double> Neuron::activate(std::vector< micrograd::Value<double> > inputs) noexcept
{
	if (inputs.size() != this->weights.size())
		return 0.0;

	micrograd::Value<double> activation = 0.0;

	std::vector< micrograd::Value<double> > products;

	for (size_t i = 0; i < inputs.size(); i++)
		products.push_back(inputs[i] * this->weights[i]);

	for (size_t i = 0; i < inputs.size(); i++)
		activation += products[i];

	activation += this->bias;

	return activation.tanh();
}