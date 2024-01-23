
#include <iostream>
#include "tensor.h"
#define N 1024 * 2

using namespace tensor;

// Structure from : http://www.flounder.com/cpuid_explorer2.htm

// Bit fields and unions : https://icarus.cs.weber.edu/~dab/cs1410/textbook/5.Structures/unions.html

struct eax1_intel {
	unsigned stepping_id: 4;
	unsigned model_id: 4;
	unsigned family_id: 4;
	unsigned processor_type: 2;
	unsigned : 2; // Reserved
	unsigned extended_model: 4;
	unsigned extended_family: 8;
	unsigned : 4;
};

struct ebx1_intel {
	unsigned brand_index: 8;
	unsigned CGLUSH_line_size: 8;
	unsigned logical_processors: 8;
	unsigned initial_APICID: 8;
};

struct ecx1_intel {
	unsigned SSE: 1;
	unsigned : 2;
	unsigned MWAIT: 1;
	unsigned CPL: 1;
	unsigned VMX: 1;
	unsigned : 1;
	unsigned EST: 1;
	unsigned TM2: 1;
	unsigned : 1;
	unsigned L1: 1;
	unsigned : 1;
	unsigned : 1;
	unsigned CMPXCHG16B: 1;
	unsigned : 18;
};

struct edx1_intel {
	unsigned FPU: 1;  // FPU87 on chip
	unsigned VME: 1;  // Virtual 8086 extensions
	unsigned DE : 1;  // Debugging extensions
	unsigned PSE: 1;  // Page Size extensions
	unsigned MSR: 1;  // RDMSR / WRMSR support
	unsigned TSC: 1;  // TimeStamp Counter
	unsigned PAE: 1;  // Physical Address Exception
	unsigned MCE: 1;  // Machine Check Exception
	unsigned CX8: 1;  // CMXCHG8B
	unsigned APIC: 1; // APIC on-chip
	unsigned : 1;
	unsigned SEP: 1;  // SYSENTER / SYSEXIT
	unsigned MTRR: 1; // Memory Type Range Registers
	unsigned PGE: 1;  // Global PTE Bit
	unsigned MCA: 1;  // Machine Check Architecture
	unsigned CMOV: 1; // CMOV: Conditional move/compare instruction
	unsigned PAT: 1;  // Page Attribute Table
	unsigned PSE36: 1;//PSE36 Sizr Extension
	unsigned PSN: 1;  // Processor Serial Number
	unsigned CFLSH: 1;// CFLUSH instruction
	unsigned : 1;
	unsigned DS : 1;  // Debug Store
	unsigned ACPI: 1; // ACPI
	unsigned MMX: 1;  // MMX
	unsigned FXSR: 1;
	unsigned SSE: 1;  // SSE extenstions
	unsigned SSE2: 1; // SSE2 extensions
	unsigned SS : 1;  // Self-Snoop
	unsigned HTT: 1;  // Hyperthreading
	unsigned TM : 1;  // Thermal Monitor
	unsigned : 1;    
	unsigned PBE: 1; // Pending Break Enable
};

typedef union {
	eax1_intel bits;
	uint32_t eax;
} EAX1b;

typedef union {
	ebx1_intel bits;
	uint32_t ebx;
} EBX1b;

typedef union {
	edx1_intel bits;
	uint32_t ebx;
} EDX1b;


int main(int argv, char* argc[]) {
	std::unique_ptr<float[]> data = std::make_unique<float[]>(N*N); 
	std::unique_ptr<float[]> dataint = std::unique_ptr<float[]>(new float[3840*2160*4*60]);
	Tensor<float> a = Tensor<float>(data, N*N, {N, N});
	Tensor<float> b = Tensor<float>(data, N*N, {2, 2, N/2, N/2});
	Tensor<float> d = Tensor<float>(dataint, 3840*2160*4*60, {60, 3840, 2160, 4});

	auto c = Tensor<float>::dot(a, b);

	std::cout << a << std::endl;
	std::cout << b << std::endl;
	std::cout << d << std::endl;

	EAX1b eax;
	eax.eax = c[0];

	std::cout << std::hex;
	std::cout << c[0] << std::endl;
	std::cout << eax.bits.stepping_id << std::endl;
	std::cout << eax.bits.model_id << std::endl;
	std::cout << eax.bits.family_id << std::endl;
	std::cout << eax.bits.processor_type << std::endl;

	// binary
	for( auto b = c[0]; b; b >>= 1) std::cout << (b & 1); 
	std::cout << '\n';

	uint32_t model;
	uint32_t family;
	if(eax.bits.family_id == 0x06 || eax.bits.family_id == 0x0f) {
		// extended model
		model = eax.bits.model_id + (eax.bits.extended_model << 4);
	} else {
		model = eax.bits.model_id;
	}

	if(eax.bits.family_id == 0x0f) {
		family = eax.bits.family_id + eax.bits.extended_family;
	} else {
		family = eax.bits.family_id;
	}

	std::cout << model << std::endl;
	std::cout << family << std::endl;
}

