
#include<limits.h>

#define TENSOR_MAX_STORAGE_SIZE UINT_MAX

namespace tensorlib {

enum Device {
	GPU,
	CPU
};

enum {
	SUCCESSFUL,
	INVALID_ARGUMENTS,
	WRONG_NUMBER_OF_ELEMENTS,
	GLOBAL_LIMIT_EXCEEDED,
	UNEXPECTED ERROR,
} movement_op_ret;

struct View {
	std::shared_ptr<uint32_t[]> view;
	std::shared_ptr<uint32_t[]> strides;
	uint32_t ndim;
	uint64_t telem;
};

template<typename T>
struct Tensor {
	std::shared_ptr<T[]> storage;
	std::shared_ptr<View> shape;
	std::unique_ptr<Tensor<float>> grad;
	Device device;
	uint64_t size;
	bool bgrad;
};

void restride(View& v) {
	v.strides = std::make_unique<uint32_t[]>(v.ndim);
	for(size_t i=0; i < v.ndim; i++) { v.strides[i] = 1;	}
	for(size_t i=v.ndim; i > 0; i--) {
		if(i == v.ndim) continue;
		v.strides[i-1] = v.strides[i] * v.view[i];
	}
}

View create_view(std::initializer_list<uint32_t> argview) {
	View ret;
	uint8_t i = 0;
	uint64_t sum = 1;
	ret.view = std::make_unique<uint32_t[]>(argview.size());
	for(const auto& x : argview) {
		ret.view[i] = x;
		sum *= x;
		i++;
	}
	if(sum > TENSOR_MAX_STORAGE_SIZE) { throw std::runtime_error("Number of elements in Tensor exceeds TENSOR_MAX_STORAGE_SIZE"); }
	ret.telem = sum;
	ret.ndim = argview.size();
	restride(ret);
	return ret;
}

movement_op_ret reshape(View v, std::shared_ptr<uint32_t[]>& argshape, size_t& len) {
	uint64_t product=1;
	for(size_t i=0; i<len; i++) { product *= argview[i]};
	if(product != v.telem) return INVALID_ARGUMENTS;
	v.ndim = len;
	v.view = std::make_unique<uint32_t[]>(len);
	for(size_t i=0; i<len; i++) {
		v.view[i] = argview[i];
	}
	restride(v);
	return SUCCESS;
}

movement_op_ret permute(View v, std::shared_ptr<uint32_t[]>& idxs, size_t& len) {

}

template><typename T>
Tensor create_tensor(std::shared_ptr<T[]> data, uint64_t size, 
								     std::initializer_list<uint32_t>shape, bool grad=false,
										 Device device=GPU) {

}

}; // namespace
