#include "stdafx.h"
#include "Layer.h"

Layer::Layer(unsigned numOfInputs, unsigned numOfOutputs) noexcept
    : final_sum(0.0)
{
    this->neurons.reserve(numOfOutputs);
    
    for (unsigned i = 0; i < numOfOutputs; i++)
        this->neurons.push_back(Neuron(numOfInputs));
}

micrograd::Value<double> Layer::activate(std::vector<micrograd::Value<double>>& inputs) noexcept
{
    this->activations.clear();
    this->activations.reserve(inputs.size() * 1.5);

    for (size_t i = 0; i < this->neurons.size(); i++)
        this->activations.push_back(this->neurons[i](inputs));

    if (this->activations.size() % 2 != 0)
        this->activations.push_back(micrograd::Value<double>(0.0));

    const size_t before_size = this->activations.size();

    for (size_t i = 0; i + 1 < before_size; i += 2)
        this->activations.push_back(this->activations[i] + this->activations[i + 1]);

    this->final_sum = this->activations[this->activations.size() - 1];
}

micrograd::Value<double> Layer::operator()(std::vector<micrograd::Value<double>>& inputs) noexcept
{
    return this->activate(inputs);
}
