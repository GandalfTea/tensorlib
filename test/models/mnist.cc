
#include "tensor.h"
#include <cassert>

class Linear() {
	std::unique_ptr<Tensor<float> weights = nullptr;
	std::unique_ptr<Tensor<float> bias = nullptr;
	uint32_t params;
	bool bbias;

	Linear(size_t fan_in, size_t fan_out, bool bias=true) 
		: bias(bias), params(fan_in*fan_out)
	{
		weights = std::unique_ptr<Tensor<float>>(Tensor<float>::randn({fan_out, fan_in}));	
		if(bbias) weights = std::unique_ptr<Tensor<float>>(Tensor<float>::randn({fan_out));	
	}

	Tensor<float> operator( Tensor<float> x) {
		assert(x.shape()[x.ndim()-1] == fan_in);
		return x.dot(weights) + bias;
	}
};


class TinyNN() {
	std::unique_ptr<Linear>) l1;
	std::unique_ptr<Linear>) l2;

	TinyNN() {
		this->l1 = std::unique_ptr<Linear>( Linear(782, 128));
		this->l2 = std::unique_ptr<Linear>( Linear(128, 10));
	}

	Tensor<float> forward( Tensor<float> x) {
		return l2(l1(x).relu()).softmax();
	}

	uint32_t parameters() { return l1.params + l1.params;	}
}

int main() {
	auto net = TinyNN();
}
