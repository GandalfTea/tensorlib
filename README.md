
Lightweight header-only tensor library for C++17 and later.

&nbsp;

### Empty initialisation
Create a virtual tensor and allocate the memory:
```c++
auto a = Tensor<float>({1024, 1024}).allocate();
```
All the memory allocated by the library is aligned according to the macro `DATA_ALIGNMENT` defaulted to 32 bytes for AVX-256, unless the size of the block is smaller then the alignment requirement.

&nbsp;


### Data initialisation

Constructors accept `T*` to allocated data. If the macro `FORCE_ALIGNMENT` is defined, values will be copied into an aligned memory block. Otherwise, all memory blocks passed into tensor constructors must be `malloc`ed because garbage collection is handled with `free()`. The shape can be either an `initializer_list` or an array.
```c++
Tensor(T* data, uint64_t size, std::initializer_list<uint32_t> shp, bool grad=false, Device dev=CPU)
Tensor(T* data, uint64_t size, uint32_t* shape, size_t slen, bool grad=false, Device dev=CPU)
```
```c++
float* data = static_cast<float*>( aligned_alloc(32, 1024*1024*sizeof(float)) );
auto a = Tensor<float>(data, 1024*1024, {1024/2, 2, 1024});
```

&nbsp;

Elements can also be entered manually using an `std::initializer_list<T>` in which case there is no `size` argument:
```c++
auto a = Tensor<float>({1, 2, 3, 4, 5}, {2, 3});
auto b = Tensor<float>({1, 2, 3, 4, 5}, s); // sized_array
```

&nbsp;

### Helpers and Getters
```c++
auto a = Tensor<float>(data, 2048*2048, {2, 1024, 2048});
```
* `a.ndim()` returns number of dimensions.
* `a.numel()` returns the number of elements in the tensor.
* `a.disklen()` returns the number of elements stored in memory.
* `a.bgrad` returns a `bool` that tracks if the tensor keeps a gradient array.
* `a.bresolved` returns a `bool` that tracks if the view and `numel` values all exist in the underlying disk data.
* `a.is_initialized` returns a `bool` that tracks if underlying tensor data exists.   
       
Modifying the following can destroy the internal integrity of the tensor object:
* `a.data()` returns a `std::shared_ptr<T[]>` to the underlying tensor data.
* `a.view()` returns a `std::shared_ptr<uint32_t[]>` to the tensor view data.
* `a.strides()` returns a `std::shared_ptr<uint32_t[]>` to the tensor stride data.

&nbsp;

### Construction helpers
Help construct special tensors. Throw if tensor data already exists.

&nbsp;

* `fill` : fill a tensor with one value.
```c++
auto a = Tensor<float>::fill({2048, 2048}, 1.0);
auto b = Tensor<float>({2048, 2048}).fill(0.0);
auto c = Tensor<>::like(a).fill(1.f); // Shallow copy and fill
```
This only stores one value in memory and redirects to it given all correct indices. Because the view does not correspond to contiguous memory, data can only be accessed using `operator()` and `bresolved` will return `false`. Manually accessing the `data()` array will yield random values or errors.

&nbsp;

* `arange` : fill a tensor with evenly spaced values in a given interval.
```c++
static Tensor<T> arange(T stop, T start=0, T step=1, Device device=CPU) {
```
```c++
auto a = Tensor<float>::arange(50);
auto b = Tensor<float>::arange(50, 10);
auto c = Tensor<float>::arange(50, 10, 5);
```
Stores all resulting values in memory, `bresolved` is `true`.

&nbsp;


* `randn` : fill a tensor with randomly generated values that fit a given distribution.
      
Distributions can be either `NORMAL`, `UNIFORM` or `CHI_SQUARED`.

```c++
void randn(T up=1.f, T down=0.f, uint32_t seed=0, Distribution dist=NORMAL);
static Tensor<T> randn(std::initializer_list<uint32_t> shp, T up=1.f, T down=0.f,
                       uint32_t seed=0, Device device=CPU, Distribution dist=NORMAL);
```
```c++
auto a = Tensor<float>::randn({2048, 2048}, 3.14, -3.14);
auto b = Tensor<>::like(a).randn(3.14); // Shallow copy and fill with randn()
```
Stores all resulting values in memory, `bresolved` is `true`.     
       
You can also randomly generate arrays of the same distributions:
```c++
static std::unique_ptr<float[]> f32_generate_uniform_distribution(uint32_t count, float up=1.f,
                           float down=0.f, double seed=0, bool bepsilon=false, float epsilon=0);

static std::unique_ptr<float[]> f32_generate_chi_squared_distribution(uint32_t count,
                            float up=1.f, float down=0.f, double seed=0);

static std::unique_ptr<float[]> f32_generate_box_muller_normal_distribution(uint32_t count,
                           float up=1.f, float down=0.f, double seed=0);
```
```c++
auto a = Tensor<>::f32_generate_uniform_distribution(2048*2048, 3.14, -3.14);
auto b = Tensor<>::f32_generate_chi_squared_distribution(2048*2048, 3.14, -3.14);
auto c = Tensor<>::f32_generate_box_muller_normal_distribution(2048*2048, 3.14, -3.14);
```

&nbsp;


* `eye`: creates an identity tensor.
```c++
static Tensor<T> eye(uint32_t size, uint32_t dims=2, Device device=CPU);
```
```c++
auto a = Tensor<>::eye(4096, 4);
auto b = Tensor<>::like(a).eye(); // Not yet implemented
```
Values are generated when indexing, `bresolved` is `false`.

&nbsp;

### Virtual tensors     
Tensors that have no underlying data.
```c++
auto a = Tensor<float>({2048, 2048});
```
```c++
sized_array<uint32_t> s { std::unique_ptr<uint32_t[]>(new uint32_t[2]()), 2};
s.ptr[0] = 2;
s.ptr[1] = 3;
Tensor<float> b(s);
```

&nbsp;


### Movement Operations
They perform changes to the `view` and `stride` of the tensor, but make no changes to the underlying data.


&nbsp;

* `reshape`: change the `view` of the tensor without changing the number of elements inside.
```c++
auto a = Tensor<float>(data, 2048*2048, {2048, 2048});
a.reshape({2, 1024, 2048});
a.reshape({1024, 2, 1024, 2});
```

&nbsp;

* `permute`: change the order of the dimensions inside of the tensor.
```c++
auto a = Tensor<float>(data, 2048*2048, {1024, 2, 1024, 2});
a.permute({1, 3, 0, 2}) // {2, 2, 1024, 1024}
```
