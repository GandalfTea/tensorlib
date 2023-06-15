
#ifndef TENSOR
#define TENSOR

#define TENSOR_MAX_DIM 2 << 15
#define TENSOR_MAX_STORAGE_SIZE 2 << 63 

#include <array>
#include <cassert>
#include <initializer_list>

using std::size_t;

namespace tensor {

enum MovementOps {
	RESHAPE,
	PERMUTE,
	EXPAND,
	PAD,
	SHRINK,
	STRIDE
};

template<uint32_t N>
struct View {
	std::array<uint32_t, N> view;
	std::array<uint32_t, N> strides;

	Void(std::initializer_list<uint32_t> argview) {
		assert(sizeof(argview) == N);
		uint8_t i = 0;
		for(const auto& x : argview) {
			this->view[i] = x;
			i++;
		}
		this->calculate_strides(this->view);
	}

	private:
		void calculate_strides(std::array<uint32_t, N> view) {
			std::array<uint32_t, N> tmp;
			for(size_t i=N-1; i >= 0; i--) {
				if(i==N-1) tmp[N-1] = 1;	
				tmp[i] = tmp[i+1] * view[i+1];
			}	
			this-> strides = tmp;
		}
};

template<uint32_t M>
class ShapeTracker {
	public:
		struct View<M> view;
		size_t size;
};

template<typename T, size_t N, size_t M>
class Tensor {
	public:
		Tensor(T* arr, std::initializer_list<uint32_t> shape) 
			: storage(*arr), storage_size(*arr.size())
		{ };

		uint32_t* get_shape() {
			return this->shape.view;
		}

		bool reshape(std::initializer_list<uint32_t> nview) {
			if(this->is_valid_view(nview)) {
				this->shape.view = nview;
				return 1;
			}
			return 0;
		}

	private:
		uint64_t storage_size;
		std::array<T, N> storage;
		View<M> shape;

		bool is_valid_view(std::initializer_list<uint32_t> shape) {
			uint32_t p = 1;
			for(const uint32_t& i : shape) { p *= i; }
			if(this->storage_size % p == 0) return 1;	
			else return 0;
		}
};


// uint32_t arr[] = { . . . };
// auto a = Tensor<uint32_t, arr.size()>( arr, {2, 2, -1} );
// auto a = Tensor<typeof arr, arr.size()>(arr, {2, 2, -1});

} // namespace
#endif
