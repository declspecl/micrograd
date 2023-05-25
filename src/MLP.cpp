#include "stdafx.h"

#include "MLP.h"

MLP::MLP(std::vector<unsigned> layerSizes) noexcept
{
	this->layers.reserve(layerSizes.size() - 1);

	for (unsigned i = 0; i + 1 < layerSizes.size(); i++)
		this->layers.push_back(Layer(layerSizes[i], layerSizes[i + 1]));
}

std::vector< micrograd::Value<double> > MLP::activate(std::vector< micrograd::Value<double> >& inputs) noexcept
{
	this->forward_passes = { inputs };
	this->forward_passes.reserve(this->layers.size());

	for (Layer& layer : this->layers)
		this->forward_passes.push_back(layer.activate(this->forward_passes[this->forward_passes.size() - 1]));

	return this->forward_passes[this->forward_passes.size() - 1];
}

std::vector< micrograd::Value<double> > MLP::operator()(std::vector< micrograd::Value<double> >& inputs) noexcept
{
	return this->activate(inputs);
}
