
#ifndef TENSOR
#define TENSOR

#define TENSOR_MAX_DIM 2 << 15
#define TENSOR_MAX_STORAGE_SIZE 2 << 63 

#include <array>

using std::size_t;

namespace tensor {

struct View {
	uint32_t* view;
};

template<size_t N>
class ShapeTracker {
	public:
		struct View<N> view;
		size_t size;
};

template<typename T, size_t N>
class Tensor {
	public:
		Tensor(T* arr, std::initialiser_list<uint32_t> shape) 
			: storage(*arr)
		{ };

		uint32_t* shape() {
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
		View shape;

		bool is_valid_view(std::initializer_list<uint32_t> shape) {
			uint16_t p = 1;
			for(const auto& i : *shape) { p *= *i; }
			if(this->storage_size % p == 0) return 1;	
			else return 0;
		}
};


// uint32_t arr[] = { . . . };
// auto a = Tensor<uint32_t, arr.size()>( arr, {2, 2, 1} );

} // namespace
#endif
