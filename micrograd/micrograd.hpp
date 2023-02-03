#pragma once

#include <math.h>
#include <string>
#include <format>
#include <vector>
#include <algorithm>

#include <iostream>

namespace micrograd
{
	constexpr double h = 0.0001;

	template <typename T>
	double difference_quotient(double (*f)(T x), T x, double _h = h)
	{
		return ((*f)(x + _h) - (*f)(x)) / _h;
	}

	template <typename T>
	class Value
	{
	private:
		void propagateChildren();

	public:
		T data;
		double grad;

		char op;
		std::vector< Value<T>* > children;

		Value() = delete;
		Value(T data);
		Value(T data, double grad, char op, std::vector< Value<T>* > children);
		Value(const Value<T>& other);
		Value(Value<T>&& other);

		void backPropagate();

		std::string toString();

		Value<T> operator+(Value<T>& other);
		Value<T> operator-(Value<T>& other);
		Value<T> operator*(Value<T>& other);
		Value<T> operator/(Value<T>& other);
		Value<T> tanh();
	};

	template <typename T>
	Value<T>::Value(T data)
		: data(data)
		, grad(0.0)
		, op('_')
		, children({})
	{

	}

	template <typename T>
	Value<T>::Value(T data, double grad, char op, std::vector< Value<T>* > children)
		: data(data)
		, grad(grad)
		, op(op)
		, children(children)
	{

	}

	template <typename T>
	Value<T>::Value(const Value<T>& other)
		: data(other.data)
		, grad(other.grad)
		, op(other.op)
		, children(other.children)
	{

	}

	template <typename T>
	Value<T>::Value(Value<T>&& other)
		: data(std::move(other.data))
		, grad(std::move(other.grad))
		, op(std::move(other.op))
		, children(std::move(other.children))
	{

	}

	template <typename T>
	void Value<T>::propagateChildren()
	{
		switch (this->children.size())
		{
		case 2:
			switch (this->op)
			{
			case '+':
				this->children[0]->grad += this->grad;
				this->children[1]->grad += this->grad;
				break;

			case '*':
				this->children[0]->grad += this->grad * this->children[1]->data;
				this->children[1]->grad += this->grad * this->children[0]->data;
				break;

			case '-':
				this->children[0]->grad += this->grad;
				this->children[1]->grad += -this->grad;
				break;

			case '/':
				this->children[0]->grad += this->grad * (1.0 / this->children[1]->data);
				this->children[1]->grad += this->grad * -1.0 * (this->children[0]->data / (this->children[1]->data * (this->children[1]->data + h)));
				break;

			default:
				break;
			}
			break;

		case 1:
			// squish
			switch (this->op)
			{
			case 'T':
				this->children[0]->grad += 1.0 - (this->data * this->data);
				break;

			default:
				break;
			}

		case 0:
		default:
			break;
		}
	}

	template <typename T>
	void Value<T>::backPropagate()
	{
		this->grad = 1.0;

		std::vector< Value<T>* > queue = { this };

		for (size_t l = 0; l < queue.size(); l++)
		{
			Value<T>*& curr = queue[l];

			curr->propagateChildren();

			for (Value<T>* child : curr->children)
				queue.push_back(child);
		}
	}

	template <typename T>
	std::string Value<T>::toString()
	{
		std::string childrenString = "{";

		for (Value<T>* child : this->children)
			childrenString += std::to_string(child->data) + ", ";

		if (this->children.size() > 0)
		{
			childrenString.pop_back();
			childrenString.pop_back();
		}

		childrenString += "}";

		return std::format("value: {}, grad: {}, op: {}, children: {}",
			this->data, this->grad, this->op, childrenString);
	}

	template <typename T>
	Value<T> Value<T>::operator+(Value<T>& other)
	{
		return Value<T>(this->data + other.data, 0.0, '+', { this, &other });
	}

	template <typename T>
	Value<T> Value<T>::operator-(Value<T>& other)
	{
		return Value<T>(this->data - other.data, 0.0, '-', { this, &other });
	}

	template <typename T>
	Value<T> Value<T>::operator*(Value<T>& other)
	{
		return Value<T>(this->data * other.data, 0.0, '*', { this, &other });
	}

	template <typename T>
	Value<T> Value<T>::operator/(Value<T>& other)
	{
		return Value<T>(this->data / other.data, 0.0, '/', { this, &other });
	}

	template <typename T>
	Value<T> Value<T>::tanh()
	{
		return Value<T>((::exp(2 * this->data) - 1) / (::exp(2 * this->data) + 1), 0.0, 'T', {this});
	}
}