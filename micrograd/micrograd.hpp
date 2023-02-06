#pragma once

#include <math.h>
#include <string>
#include <format>
#include <vector>
#include <algorithm>

#include <iostream>

namespace micrograd
{
	namespace constants
	{
		constexpr double h = 0.0001;
		constexpr double pi = 3.141592653589;
	}

	template <typename T>
	class Value
	{
	private:
		// internal function to calculate gradients for children
		void propagateChildren() noexcept;

	public:
		// instance data
		T data;
		double grad;

		char op;
		std::vector< Value<T> > children;

		// base constructors
		Value() = delete;
		Value(T data) noexcept;
		Value(T data, double grad, char op, std::vector< Value<T> > children) noexcept;

		// copy & move constructors
		Value(const Value<T>& other) noexcept;
		Value(Value<T>&& other) noexcept;

		// copy & move assignment operator overloads
		Value<T>& operator=(const Value<T>& other) noexcept;
		Value<T>& operator=(Value<T>&& other) noexcept;

		// backpropagation
		void backPropagate() noexcept;
		void zeroGrad() noexcept;

		// get all children recursively
		std::vector< Value<T>* > parameters() noexcept;

		// utility functions
		std::string toString() const noexcept;

		// implementing operations
		Value<T> operator+(const Value<T>& other) noexcept;
		Value<T> operator-(const Value<T>& other) noexcept;
		Value<T> operator*(const Value<T>& other) noexcept;
		Value<T> operator/(const Value<T>& other) noexcept;

		Value<T>& operator+=(const Value<T>& other) noexcept;
		Value<T>& operator-=(const Value<T>& other) noexcept;
		Value<T>& operator*=(const Value<T>& other) noexcept;
		Value<T>& operator/=(const Value<T>& other) noexcept;

		Value<T> tanh() noexcept;
	};

	template <typename T>
	Value<T>::Value(T data) noexcept
		: data(data)
		, grad(0.0)
		, op('_')
		, children({})
	{

	}

	template <typename T>
	Value<T>::Value(T data, double grad, char op, std::vector< Value<T> > children) noexcept
		: data(data)
		, grad(grad)
		, op(op)
		, children(children)
	{

	}

	template <typename T>
	Value<T>::Value(const Value<T>& other) noexcept
		: data(other.data)
		, grad(other.grad)
		, op(other.op)
		, children(other.children)
	{

	}

	template <typename T>
	Value<T>::Value(Value<T>&& other) noexcept
		: data(std::move(other.data))
		, grad(std::move(other.grad))
		, op(std::move(other.op))
		, children(std::move(other.children))
	{

	}

	template <typename T>
	Value<T>& Value<T>::operator=(const Value<T>& other) noexcept
	{
		this->data = other.data;
		this->grad = other.grad;
		this->op = other.op;
		this->children = other.children;

		return *this;
	}

	template <typename T>
	Value<T>& Value<T>::operator=(Value<T>&& other) noexcept
	{
		this->data = std::move(other.data);
		this->grad = std::move(other.grad);
		this->op = std::move(other.op);
		this->children = std::move(other.children);

		other.data = 0x00;
		other.grad = 0x00;
		other.op = 0x00;
		other.children = {};

		return *this;
	}

	template <typename T>
	void Value<T>::propagateChildren() noexcept
	{
		for (Value<T>& child : this->children)
			child.grad = 0.0;

		switch (this->children.size())
		{
		case 2:
			switch (this->op)
			{
			case '+':
				this->children[0].grad += this->grad;
				this->children[1].grad += this->grad;
				break;

			case '*':
				this->children[0].grad += this->grad * this->children[1].data;
				this->children[1].grad += this->grad * this->children[0].data;
				break;

			case '-':
				this->children[0].grad += this->grad;
				this->children[1].grad += -this->grad;
				break;

			case '/':
				this->children[0].grad += this->grad * (1.0 / this->children[1].data);
				this->children[1].grad += this->grad * -1.0 * (this->children[0].data / (this->children[1].data * (this->children[1].data + constants::h)));
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
				this->children[0].grad += 1.0 - (this->data * this->data);
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
	void Value<T>::backPropagate() noexcept
	{
		this->grad = 1.0;

		std::vector< Value<T>* > queue = { this };

		for (size_t l = 0; l < queue.size(); l++)
		{
			Value<T>*& curr = queue[l];

			curr->propagateChildren();

			for (Value<T>& child : curr->children)
				queue.push_back(&child);
		}
	}

	template <typename T>
	void Value<T>::zeroGrad() noexcept
	{
		this->grad = 0.0;

		std::vector< Value<T>* > queue = { this };

		for (size_t l = 0; l < queue.size(); l++)
		{
			Value<T>*& curr = queue[l];

			curr->grad = 0.0;

			for (Value<T>& child : curr->children)
				queue.push_back(&child);
		}
	}

	template <typename T>
	std::vector< Value<T>* > Value<T>::parameters() noexcept
	{
		std::vector< Value<T>* > queue = { this };

		for (size_t l = 0; l < queue.size(); l++)
			for (Value<T>& child : queue[l]->children)
				queue.push_back(&child);

		return queue;
	}

	template <typename T>
	std::string Value<T>::toString() const noexcept
	{
		std::string childrenString = "{";

		for (const Value<T>& child : this->children)
			childrenString += child.toString() + ", ";

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
	Value<T> Value<T>::operator+(const Value<T>& other) noexcept
	{
		return Value<T>(this->data + other.data, 0.0, '+', { *this, other });
	}

	template <typename T>
	Value<T> Value<T>::operator-(const Value<T>& other) noexcept
	{
		return Value<T>(this->data - other.data, 0.0, '-', { *this, other });
	}

	template <typename T>
	Value<T> Value<T>::operator*(const Value<T>& other) noexcept
	{
		return Value<T>(this->data * other.data, 0.0, '*', { *this, other });
	}

	template <typename T>
	Value<T> Value<T>::operator/(const Value<T>& other) noexcept
	{
		return Value<T>(this->data / other.data, 0.0, '/', { *this, other });
	}

	template <typename T>
	Value<T>& Value<T>::operator+=(const Value<T>& other) noexcept
	{
		*this = Value<T>(this->data + other.data, 0.0, '+', { *this, other });

		return *this;
	}

	template <typename T>
	Value<T>& Value<T>::operator-=(const Value<T>& other) noexcept
	{
		*this = Value<T>(this->data - other.data, 0.0, '-', { *this, other });

		return *this;
	}

	template <typename T>
	Value<T>& Value<T>::operator*=(const Value<T>& other) noexcept
	{
		*this = Value<T>(this->data * other.data, 0.0, '*', { *this, other });

		return *this;
	}

	template <typename T>
	Value<T>& Value<T>::operator/=(const Value<T>& other) noexcept
	{
		*this = Value<T>(this->data / other.data, 0.0, '/', { *this, other });

		return *this;
	}

	template <typename T>
	Value<T> Value<T>::tanh() noexcept
	{
		return Value<T>((::exp(2 * this->data) - 1) / (::exp(2 * this->data) + 1), 0.0, 'T', { Value<T>(*this) });
	}
}