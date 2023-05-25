#pragma once

#include "constants.h"

#include <tuple>
#include <math.h>
#include <string>
#include <format>
#include <memory>
#include <vector>

#ifdef _DEBUG
#include <iostream>
#endif

namespace micrograd
{
    class Value;

    typedef std::shared_ptr<Value> SharedValue;

	class Value
    {
	private:
		// internal function to calculate gradients for children
		void grad_children() noexcept;

	public:
		// instance data
		double data;
		double grad;

		char op;
		std::tuple<SharedValue, SharedValue> children;

		// base constructors
		Value(double data) noexcept;
		Value(double data, double grad, char op, std::tuple<SharedValue, SharedValue> children) noexcept;

		// copy & move methods
		Value(const Value& other) noexcept;
        Value operator=(const Value& other) noexcept;

		Value(Value&& other) noexcept;
		Value operator=(Value&& other) noexcept;

		// backpropagation
		void back_prop() noexcept;
		void zero_grad() noexcept;
		std::vector<SharedValue> parameters() noexcept;

		// utility functions
		std::string to_string() const noexcept;

		// mathematical operations
		Value operator+(const Value& other) noexcept;
		Value operator-(const Value& other) noexcept;
		Value operator*(const Value& other) noexcept;
		Value operator/(const Value& other) noexcept;

		Value tanh() noexcept;

		// deleted functions
		Value() = delete;
	};
}
