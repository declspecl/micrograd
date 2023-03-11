#pragma once

#include "Layer.h"

#include <vector>

class MLP
{
private:
	std::vector< std::vector< micrograd::Value<double> > > forward_passes;

public:
	std::vector<Layer> layers;

	MLP(std::vector<unsigned> layerSizes) noexcept;

	std::vector< micrograd::Value<double> > activate(std::vector< micrograd::Value<double> >& inputs) noexcept;
	std::vector< micrograd::Value<double> > operator()(std::vector< micrograd::Value<double> >& inputs) noexcept;
};