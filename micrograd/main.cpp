#include "micrograd.hpp"

#include <stdio.h>

using namespace micrograd;

int main()
{
	Value<double> a = -2;
	Value<double> b = 3;

	Value d = a * b;
	Value e = a + b;
	Value f = d * e;

	f.backPropagate();

	printf("a | %s\n", a.toString().c_str());
	printf("b | %s\n", b.toString().c_str());
	printf("d | %s\n", d.toString().c_str());
	printf("e | %s\n", e.toString().c_str());
	printf("f | %s\n", f.toString().c_str());

	return 0;
}