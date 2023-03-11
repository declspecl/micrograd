# micrograd

![awww](assets/puppy.jpg)

Micrograd is a small autograd engine in C++ that I made while watching [Andrej Karpathy's introduction to neural networks](https://youtu.be/VMj-3S1tku0) (based on his autograd engine [micrograd](https://github.com/karpathy/micrograd). It can create a Multi Level Perceptron and it supports backpropagation through various mathematical operations. I am super interested in neural networks, machine learning, and AI in general, and Karpathy's series has been hugely beneficial for me. I am not finished with his entire series yet, but I am always trying to make time for it!

This is a C++ implementation of it rather than Python which, naturally, works a bit differently. In Python, due to its garbage collection, creating isolated Value objects without managing their lifecycle is a huge advantage, as even simple operations like summation become much more complex when every intermediate sum value needs to be carefully managed as to avoid memory leaks. I also wanted to challenge myself to not used reference counted smart pointers because although it would still, of course, work differently than Karpathy's original micrograd, using std::shared_ptr would mimic a lot of the behavior in his version, and I thought it would be a fun challenge to dance around with the program to try to support the same functionality. Additionally, design choices like including a copy constructor are more difficult decisions to make because now, multiple instances of Value point to the same Value object, so the backpropagation may mutate objects that are being used by other instances as well.

# Installation
Run `git clone https://github.com/declspecl/micrograd.git` to download the repository. Then, open micrograd.sln in Visual Studio, and it should build right of the bat! Included in the /micrograd folder is a demo.cpp file that shows briefly how it can be used.

# Features
- Backpropagation
- Value, Neuron, Layer, and full Multi Layer Perceptron classes
- Support for addition, subtraction, multiplication, division and tanh
- Recursive string representation in the form of Value.to_string()
