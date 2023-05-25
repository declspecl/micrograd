#include "stdafx.h"

#include "Layer.h"

Layer::Layer(unsigned numOfInputs, unsigned numOfOutputs) noexcept
{
    this->neurons.reserve(numOfOutputs);
    
    for (unsigned i = 0; i < numOfOutputs; i++)
        this->neurons.push_back(Neuron(numOfInputs));
}

std::vector< micrograd::Value<double>* > Layer::parameters()
{
    std::vector< micrograd::Value<double>* > parameters;
    parameters.reserve(this->neurons.size() * this->neurons[0].weights.size());

    for (Neuron& neuron : this->neurons)
        for (micrograd::Value<double>* parameter : neuron.parameters())
            parameters.push_back(parameter);

    return parameters;
}

std::vector< micrograd::Value<double> > Layer::activate(std::vector<micrograd::Value<double>>& inputs) noexcept
{
    std::vector< micrograd::Value<double> > activations;
    activations.reserve(inputs.size());

    for (Neuron& neuron : this->neurons)
        activations.push_back(neuron.activate(inputs));

    return activations;
}

std::vector< micrograd::Value<double> > Layer::operator()(std::vector<micrograd::Value<double>>& inputs) noexcept
{
    return this->activate(inputs);
}
