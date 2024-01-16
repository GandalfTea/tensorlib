
Lightweight header-only tensor library for C++17 and later.

#### Create


Virtual tensors (no data)

```c++
// initializer list
auto a = Tensor<float>({2048, 2048});

// sized array 
sized_array<uint32_t> s;
s.ptr = std::unique_ptr<uint32_t[]>(new uint32_t[2]());
s.ptr[0] = 2048;
s.ptr[1] = 2048;
s.size = 2;
Tensor<float> a(s);
```

&nbsp;

Data initialization

```c++
// float array 
std::unique_ptr<float[]> data = std::unique_ptr<float[]>( new float[2048*2048]() );
auto a = Tensor<float>(data, 2048*2048, {2, 1024, 2048});
auto a = Tensor<>(data, 2048*2048, {2, 1024, 2048}); // Equivalent

// initializer list
auto a = Tensor<float>({1, 2, 3, 4, 5}, {2, 3});

// sized array shape
sized_array<uint32_t> s;
s.ptr = std::unique_ptr<uint32_t[]>(new uint32_t[2]());
s.ptr[0] = 2;
s.ptr[1] = 3;
s.size = 2;
auto a = Tensor<float>({1, 2, 3, 4, 5}, s);

std::unique_ptr<float[]> data = std::unique_ptr<float[]>( new float[2*3]() );
auto b = Tensor<b>(data, 2*3, s);
```

&nbsp;

Special initialization

&nbsp;

Fill
```c++
auto a = Tensor<float>({2048, 2048}).fill(0.0);
```

&nbsp;

arange
```c++
auto a = Tensor<float>::arange(50, 10, 5);
```

&nbsp;

randn
```c++
auto a = Tensor<float>::randn({2048, 2048}, 3.14, -3.14);
```

&nbsp;

eye
```c++
auto a = Tensor<float>::eye(2048, 2);
```
