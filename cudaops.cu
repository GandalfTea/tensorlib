
// cmake might error on lib linking first
#ifdef WINDOWS
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

enum {
	float64,
	float32,
	float16
} CUDATypes;

template<enum T>
struct CUDABuffer {
 CUDATypes type = T;
 uint64_t size = 0;
}
