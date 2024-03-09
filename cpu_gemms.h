

// sorry MSVC bros
// microsoft is baka
#if defined(__AVX__) && (defined(__FMA__) || defined(__FMA4__)) 

#include <memory>
#include <stdint.h>
#include <omp.h>
#include <immintrin.h>

#include <iostream> // debug


// bit fields

// 0x01 - extended basic information

typedef union {
  struct {
    unsigned stepping_id: 4;
    unsigned model_id: 4;
    unsigned family_id: 4;
    unsigned processor_type: 2;
    unsigned : 2; 
    unsigned extended_model: 4;
    unsigned extended_family: 8;
    unsigned : 4;
  } intel;
  struct {
    unsigned stepping_id: 4;
    unsigned model_id: 4;
    unsigned family_id: 4;
    unsigned : 4; 
    unsigned extended_model: 4;
    unsigned extended_family: 8;
    unsigned : 4;
  } amd;
  uint32_t eax;
} eax1b;

typedef union {
  struct {
    unsigned brand_index: 8;
    unsigned CGLUSH_line_size: 8;
    unsigned logical_processors: 8;
    unsigned initial_APICID: 8;
  } bits;
  uint32_t ebx;
} ebx1b;

typedef union {
  struct {
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
  } intel;
  struct {
    unsigned SSE3: 1;
    unsigned : 12;
    unsigned CMPXCHG16B: 1;
    unsigned : 18;
  } amd;
  uint32_t ecx;
} ecx1b;

typedef union {
  struct {
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
  } intel;
  struct {
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
    unsigned : 1;  
    unsigned CFLSH: 1;// CFLUSH instruction
    unsigned : 3;
    unsigned MMX: 1;  // MMX
    unsigned FXSR: 1;
    unsigned SSE: 1;  // SSE extenstions
    unsigned SSE2: 1; // SSE2 extensions
    unsigned : 1;  
    unsigned HTT: 1;  // Hyperthreading
    unsigned : 3;  
  } amd;
  uint32_t edx;
} edx1b;


// helpers

void _cpuid(long int op, uint32_t& eax, uint32_t& ebx, uint32_t& ecx, uint32_t& edx) {
  asm volatile(
    "cpuid"
    : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
    : "a"(op)
    : "cc" 
  );
}

uint32_t cpuid_highest_leaf() {
  uint32_t ret[4];
  _cpuid(0x0, ret[0], ret[1], ret[2], ret[3]);
  return ret[0];
}

// my cpu doesn't support this so it's untested
uint32_t* cpuid_cache_info() {
  uint32_t* ret = new uint32_t[4];
  ret[0] == 0x0;
  _cpuid(0x16, ret[0], ret[1], ret[2], ret[3]);
  return ret;
}

// AMD
// 0x800000005 - L1 + TLB

// 2M+4M page TLB info
typedef union {
  struct {
    unsigned inst_count: 8;
    unsigned inst_assoc: 8;
    unsigned data_count: 8;
    unsigned data_assoc: 8;
  } bits;
  uint32_t eax;
} eax5x;

// 4K page TLB info
typedef union {
  struct {
    unsigned inst_count: 8;
    unsigned inst_assoc: 8;
    unsigned data_count: 8;
    unsigned data_assoc: 8;
  } bits;
  uint32_t ebx;
} ebx5x;

// L1 data cache info
typedef union {
  struct {
    unsigned line_size: 8;
    unsigned lines_per_tag: 8;
    unsigned associativity: 8;
    unsigned cache_size: 8;
  } bits;
  uint32_t ecx;
} ecx5x;

// L1 instruction cache info
typedef union {
  struct {
    unsigned line_size: 8;
    unsigned lines_per_tag: 8;
    unsigned associativity: 8;
    unsigned cache_size: 8;
  } bits;
  uint32_t edx;
} edx5x;


uint32_t amd_highest_8xleaf() {
  uint32_t* regs = new uint32_t[4];
  _cpuid(0x80000000, regs[0], regs[1], regs[2], regs[3]);
  return regs[0];
}

uint32_t* amd_l1_cpuid() {
  uint32_t* regs = new uint32_t[4];
  if(amd_highest_8xleaf() >= 0x80000005) {
    _cpuid(0x80000005, regs[0], regs[1], regs[2], regs[3]);
    return regs;
  } else regs[0] = 0;
  return regs;
}

// AMD
// 0x800000006 - L2 + TLB
// Associativity table pg662 https://www.amd.com/content/dam/amd/en/documents/processor-tech-docs/programmer-references/24594.pdf

// L2 TLB 2M/4M
typedef union {
  struct {
    unsigned inst_count: 12; // L2 instruction TLB number of entriees for 2M-4M
    unsigned inst_assoc: 4;  // L2 instruction TLB associativity
    unsigned data_count: 12; // L2 data TLB number of entries
    unsigned data_assoc: 4;  // L2 data TLB associativity
  } bits;
  uint32_t eax;
} eax6x;

// L2 TLB 4K
typedef union {
  struct {
    unsigned inst_count: 12;
    unsigned inst_assoc: 4;
    unsigned data_count: 12;
    unsigned data_assoc: 4;
  } bits;
  uint32_t ebx;
} ebx6x;

// L2
typedef union {
  struct {
    unsigned line_size: 8;
    unsigned : 4;
    unsigned associativity: 4;
    unsigned cache_size: 16;
  } intel;
  struct ecx6_amd {
    unsigned line_size: 8;
    unsigned lines_per_tag: 4;
    unsigned associativity: 4;
    unsigned cache_size: 16;
  } amd;
  uint32_t ecx;
} ecx6x;

// L3
typedef union {
  struct {
    unsigned line_size: 8;
    unsigned lines_per_tag: 4;
    unsigned associativity: 4;
    unsigned : 2;
    unsigned cache_size: 14; // cache_size*512kB < L3 size < (cache_size+1) * 512 kB (shared)
  } bits;
  uint32_t edx;
} edx6x;

// L2 and L3 cache + TLB
uint32_t* amd_cache_cpuid() {
  uint32_t* ret = new uint32_t[4];
  ret[0] == 0x0;
  _cpuid(0x80000006, ret[0], ret[1], ret[2], ret[3]);
  return ret;
}


// Intel
// 0x02 - cache and TLB information
// TODO: make work

/*
struct cpuregs {
  uint32_t eax;
  uint32_t ebx;
  uint32_t ecx;
  uint32_t edx;
};

typedef union {
  cpuregs regs;
  unsigned char descriptors[16];
} regs2;

struct cache_data{
  bool r = false;
  uint64_t size = 0; // in bytes
  int associative = 0; 
  int line_size = 0; // in bytes
};

struct tlb_data {
  uint64_t pages = 0;
  int associative = 0;
  int entries = 0;
};

struct cpuid2_ret {
  cache_data l1_data;
  cache_data l1_ins;
  cache_data l2;
  cache_data l3;
  tlb_data ins_tlb;
  tlb_data data_tlb0;
  tlb_data data_tlb1;
  tlb_data data_tlb;
  tlb_data stlb;
  tlb_data dtlb;
  tlb_data utlb;
};

cpuid2_ret intel_cache_tlb() {
  struct cpuid2_ret ret;
  regs2 regs;
  _cpuid(0x2, regs.regs.eax, regs.regs.ebx, regs.regs.ecx, regs.regs.edx);
    for(int i=0; i<16; i++) {
      switch( regs.descriptors[i] ) {
        case 0x00: continue;
        case 0x01: ret.ins_tlb = (tlb_data){4000, 4, 32}; continue;
        case 0x02: ret.ins_tlb = (tlb_data){4000000, 0, 2}; continue;
        case 0x03: ret.data_tlb0 = (tlb_data){4000, 4, 64}; continue;
        case 0x04: ret.data_tlb0 = (tlb_data){4000000, 4, 8}; continue;
        case 0x05: ret.data_tlb1 = (tlb_data){4000000, 4, 32}; continue;
        case 0x06: ret.l1_ins = (cache_data){1, 8000, 4, 32}; continue;
        case 0x08: ret.l1_ins = (cache_data){1, 16000, 4, 32}; continue;
        case 0x09: ret.l1_ins = (cache_data){1, 32000, 4, 64}; continue;
        case 0x0a: ret.l1_data = (cache_data){1, 8000, 2, 32}; continue;
        case 0x0b: ret.ins_tlb = (tlb_data){4000, 4, 4}; continue;
        case 0x0c: ret.l1_data = (cache_data){1, 16000, 4, 32}; continue;
        case 0x0d: ret.l1_data = (cache_data){1, 16000, 4, 64}; continue;
        case 0x0e: ret.l1_data = (cache_data){1, 24000, 6, 64}; continue;
        case 0x1d: ret.l2 = (cache_data){1, 128000, 2, 64}; continue;
        case 0x21: ret.l2 = (cache_data){1, 256000, 8, 64}; continue;
        case 0x22: ret.l3 = (cache_data){1, 512000, 4, 64}; continue;
        case 0x23: ret.l3 = (cache_data){1, 1000000, 8, 64}; continue;
        case 0x24: ret.l2 = (cache_data){1, 1000000, 16, 64}; continue;
        case 0x25: ret.l3 = (cache_data){1, 2000000, 8, 64}; continue;
        case 0x29: ret.l3 = (cache_data){1, 4000000, 8, 64}; continue;
        case 0x2c: ret.l1_data = (cache_data){1, 32000, 8, 64}; continue;
        case 0x30: ret.l1_ins = (cache_data){1, 32000, 8, 64}; continue;
        case 0x40: ret.l2 = (cache_data){0, 0, 0, 0}; continue;
        case 0x41: ret.l2 = (cache_data){1, 128000, 4, 32}; continue;
        case 0x42: ret.l2 = (cache_data){1, 256000, 4, 32}; continue;
        case 0x43: ret.l2 = (cache_data){1, 512000, 4, 32}; continue;
        case 0x44: ret.l2 = (cache_data){1, 1000000, 4, 32}; continue;
        case 0x45: ret.l2 = (cache_data){1, 2000000, 4, 32}; continue;
        case 0x46: ret.l3 = (cache_data){1, 4000000, 4, 64}; continue;
        case 0x47: ret.l3 = (cache_data){1, 8000000, 8, 64}; continue;
        case 0x48: ret.l2 = (cache_data){1, 3000000, 12, 64}; continue;
        case 0x49: ret.l2 = (cache_data){1, 4000000, 16, 64}; continue; // For Intel Xeon family 0f, model 06 this is l3
        case 0x4a: ret.l3 = (cache_data){1, 6000000, 12, 64}; continue;
        case 0x4b: ret.l3 = (cache_data){1, 8000000, 16, 64}; continue;
        case 0x4c: ret.l3 = (cache_data){1, 12000000, 12, 64}; continue;
        case 0x4d: ret.l3 = (cache_data){1, 16000000, 16, 64}; continue;
        case 0x4e: ret.l2 = (cache_data){1, 6000000, 24, 64}; continue;
        case 0x4f: ret.ins_tlb = (tlb_data){4000, -1, 32}; continue; 
        case 0x50: ret.ins_tlb = (tlb_data){4000000, -1, 64}; continue; // 4kB and 2MB or 4MB
        case 0x51: ret.ins_tlb = (tlb_data){4000000, -1, 128}; continue; // 4kB and 2MB or 4MB
        case 0x52: ret.ins_tlb = (tlb_data){4000000, -1, 256}; continue; // 4kB and 2MB or 4MB
        case 0x55: ret.ins_tlb = (tlb_data){4000000, 0, 7}; continue; // 2MB or 4MB
        case 0x56: ret.data_tlb0 = (tlb_data){4000000, 4, 16}; continue; 
        case 0x57: ret.data_tlb0 = (tlb_data){4000, 4, 16}; continue; 
        case 0x59: ret.data_tlb0 = (tlb_data){4000, 0, 16}; continue; 
        case 0x5a: ret.data_tlb0 = (tlb_data){4000000, 4, 32}; continue; // 2MB or 4MB 
        case 0x5b: ret.data_tlb = (tlb_data){4000000, -1, 16}; continue; // 4MB AND 4MB pages
        case 0x5c: ret.data_tlb = (tlb_data){4000000, -1, 128}; continue; // 4MB AND 4MB pages 
        case 0x5d: ret.data_tlb = (tlb_data){4000000, -1, 128}; continue; // 4MB AND 4MB pages 
        case 0x60: ret.l1_data = (cache_data){1, 16000, 8, 64}; continue; 
        case 0x61: ret.ins_tlb = (tlb_data){4000000, 0, 48}; continue;  // 2MB or 4MB
        case 0x63: ret.data_tlb = (tlb_data){4000000, 4, 32}; continue;  // 2MB or 4MB and a separate array with 2GB pages, 4, 4ent
        case 0x64: ret.data_tlb = (tlb_data){4000, 4, 512}; continue; 
        case 0x66: ret.l1_data = (cache_data){1, 8000, 4, 64}; continue; 
        case 0x67: ret.l1_data = (cache_data){1, 16000, 4, 64}; continue; 
        case 0x68: ret.l1_data = (cache_data){1, 32000, 4, 64}; continue; 
        case 0x6a: ret.utlb = (tlb_data){4000, 8, 64}; continue; 
        case 0x6b: ret.dtlb = (tlb_data){4000, 8, 256}; continue; 
        case 0x6c: ret.dtlb = (tlb_data){4000000, 8, 128}; continue; // 2M or 4M? MB?
        case 0x6d: ret.dtlb = (tlb_data){1000000000, 0, 16}; continue; 
        // skip 0x70, 0x71, 0x72 - trace cache
        case 0x76: ret.ins_tlb = (tlb_data){4000000, 0, 8}; continue; // 2M or 4M? MB?
        case 0x78: ret.l2 = (cache_data){1, 1000000, 4, 64}; continue; 
        case 0x79: ret.l2 = (cache_data){1, 128000, 8, 64}; continue; 
        case 0x7a: ret.l2 = (cache_data){1, 256000, 8, 64}; continue; 
        case 0x7b: ret.l2 = (cache_data){1, 512000, 8, 64}; continue; 
        case 0x7c: ret.l2 = (cache_data){1, 1000000, 8, 64}; continue; 
        case 0x7d: ret.l2 = (cache_data){1, 2000000, 8, 64}; continue; 
        case 0x7f: ret.l2 = (cache_data){1, 512000, 2, 64}; continue; 
        case 0x80: ret.l2 = (cache_data){1, 512000, 8, 64}; continue; 
        case 0x82: ret.l2 = (cache_data){1, 256000, 8, 32}; continue; 
        case 0x83: ret.l2 = (cache_data){1, 512000, 8, 32}; continue; 
        case 0x84: ret.l2 = (cache_data){1, 1000000, 8, 32}; continue; 
        case 0x85: ret.l2 = (cache_data){1, 2000000, 8, 32}; continue; 
        case 0x86: ret.l2 = (cache_data){1, 512000, 4, 64}; continue; 
        case 0x87: ret.l2 = (cache_data){1, 1000000, 8, 64}; continue; 
        case 0xa0: ret.dtlb = (tlb_data){4000, 0, 32}; continue; 
        case 0xb0: ret.ins_tlb = (tlb_data){4000, 4, 128}; continue; 
        case 0xb1: ret.ins_tlb = (tlb_data){2000000, 4, 8}; continue; // or 4M, 4-way, 4 entries
        case 0xb2: ret.ins_tlb = (tlb_data){4000, 4, 64}; continue; 
        case 0xb3: ret.data_tlb = (tlb_data){4000, 4, 128}; continue;
        case 0xb4: ret.data_tlb1 = (tlb_data){4000, 4, 256}; continue;
        case 0xb5: ret.ins_tlb = (tlb_data){4000, 8, 64}; continue;
        case 0xb6: ret.ins_tlb = (tlb_data){4000, 8, 128}; continue;
        case 0xba: ret.data_tlb1 = (tlb_data){4000, 4, 64}; continue;
        case 0xc0: ret.data_tlb = (tlb_data){4000000, 4, 8}; continue; // 4kB and 4MB
        case 0xc1: ret.stlb = (tlb_data){2000000, 8, 1024}; continue; // 4kB and 2MB
        case 0xc2: ret.dtlb = (tlb_data){2000000, 4, 16}; continue; // 4kB and 2MB
        case 0xc3: ret.stlb = (tlb_data){2000000, 6, 1536}; continue; // 4kB and 2MB also 1GB pages, 4-way, 16 entries
        case 0xc4: ret.dtlb = (tlb_data){4000000, 4, 32}; continue; // 2MB and 4MB
        case 0xca: ret.stlb = (tlb_data){4000, 4, 512}; continue;
        case 0xd0: ret.l3 = (cache_data){1, 512000, 4, 64}; continue;
        case 0xd1: ret.l3 = (cache_data){1, 1000000, 4, 64}; continue;
        case 0xd2: ret.l3 = (cache_data){1, 2000000, 4, 64}; continue;
        case 0xd6: ret.l3 = (cache_data){1, 1000000, 8, 64}; continue;
        case 0xd7: ret.l3 = (cache_data){1, 2000000, 8, 64}; continue;
        case 0xd8: ret.l3 = (cache_data){1, 4000000, 8, 64}; continue;
        case 0xdc: ret.l3 = (cache_data){1, 1500000, 12, 64}; continue;
        case 0xdd: ret.l3 = (cache_data){1, 3000000, 12, 64}; continue;
        case 0xde: ret.l3 = (cache_data){1, 6000000, 12, 64}; continue;
        case 0xe2: ret.l3 = (cache_data){1, 2000000, 16, 64}; continue;
        case 0xe3: ret.l3 = (cache_data){1, 4000000, 16, 64}; continue;
        case 0xe4: ret.l3 = (cache_data){1, 8000000, 16, 64}; continue;
        case 0xea: ret.l3 = (cache_data){1, 12000000, 24, 64}; continue;
        case 0xeb: ret.l3 = (cache_data){1, 18000000, 24, 64}; continue;
        case 0xec: ret.l3 = (cache_data){1, 24000000, 24, 64}; continue;
        case 0xfe: ret = {{0,0x18,0,0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0}}; return ret; // use leaf 0x18
        case 0xff: ret = {{0,0x4,0,0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0}}; return ret; // use leaf 0x4 
        default: return ret; // unexpected error 
      }
    return ret;
  }
}
*/


// Zen2 info: https://www.7-cpu.com/cpu/Zen2.html

// L3

// TODO: only AMD for now 
uint32_t f32_max_l3() {
  uint32_t* regs = amd_cache_cpuid(); 
  edx6x edx;
  edx.edx = regs[3];
  // use to lower limit
  uint32_t l3 = edx.bits.cache_size*512*1000; // kB to B
  return l3/4; 
}

uint32_t f32_max_l2() {
  uint32_t* regs = amd_cache_cpuid(); 
  ecx6x ecx;
  ecx.ecx = regs[2];
  uint32_t l2 = ecx.amd.cache_size*1000; // kB to B
  return l2/4; 
}

uint32_t f32_max_l1() {
  uint32_t* regs = amd_l1_cpuid();
  ecx5x ecx;
  ecx.ecx = regs[2];
  uint32_t l1 = ecx.bits.cache_size*1000; // kB to B
  return l1/4; 
}


// AVX sgemm

typedef union {
  __m256 v;
  float f[8];
} m256_t;

inline void _8x8_m256_gemm(int k, const float* a, const float* b, float* c, int ldc) {

  m256_t c0007, c1017, c2027, c3037,
         c4047, c5057, c6067, c7077,
         a_vreg, b_p0_vreg;

  c0007.v = _mm256_setzero_ps();
  c1017.v = _mm256_setzero_ps();
  c2027.v = _mm256_setzero_ps();
  c3037.v = _mm256_setzero_ps();
  c4047.v = _mm256_setzero_ps();
  c5057.v = _mm256_setzero_ps();
  c6067.v = _mm256_setzero_ps();
  c7077.v = _mm256_setzero_ps();

  for(int iiiii=0; iiiii<k; iiiii++) {
    __builtin_prefetch(a+8);
    __builtin_prefetch(b+8);

    a_vreg.v = _mm256_load_ps( (float*)a );
    b_p0_vreg.v = _mm256_load_ps( (float*)b );
    a += 8;
    b += 8;

    c0007.v += a_vreg.v * b_p0_vreg.f[0];
    c1017.v += a_vreg.v * b_p0_vreg.f[1];
    c2027.v += a_vreg.v * b_p0_vreg.f[2];
    c3037.v += a_vreg.v * b_p0_vreg.f[3];
    c4047.v += a_vreg.v * b_p0_vreg.f[4];
    c5057.v += a_vreg.v * b_p0_vreg.f[5];
    c6067.v += a_vreg.v * b_p0_vreg.f[6];
    c7077.v += a_vreg.v * b_p0_vreg.f[7];
  }

  m256_t w0, w1, w2, w3, w4, w5, w6, w7;

  w0.v = _mm256_load_ps((float*)&c[0*ldc]);
  w1.v = _mm256_load_ps((float*)&c[1*ldc]);
  w2.v = _mm256_load_ps((float*)&c[2*ldc]);
  w3.v = _mm256_load_ps((float*)&c[3*ldc]);
  w4.v = _mm256_load_ps((float*)&c[4*ldc]);
  w5.v = _mm256_load_ps((float*)&c[5*ldc]);
  w6.v = _mm256_load_ps((float*)&c[6*ldc]);
  w7.v = _mm256_load_ps((float*)&c[7*ldc]);

  c0007.v = _mm256_add_ps(c0007.v, w0.v);
  c1017.v = _mm256_add_ps(c1017.v, w1.v);
  c2027.v = _mm256_add_ps(c2027.v, w2.v);
  c3037.v = _mm256_add_ps(c3037.v, w3.v);
  c4047.v = _mm256_add_ps(c4047.v, w4.v);
  c5057.v = _mm256_add_ps(c5057.v, w5.v);
  c6067.v = _mm256_add_ps(c6067.v, w6.v);
  c7077.v = _mm256_add_ps(c7077.v, w7.v);

  _mm256_store_ps( &c[0*ldc], c0007.v);
  _mm256_store_ps( &c[1*ldc], c1017.v);
  _mm256_store_ps( &c[2*ldc], c2027.v);
  _mm256_store_ps( &c[3*ldc], c3037.v);
  _mm256_store_ps( &c[4*ldc], c4047.v);
  _mm256_store_ps( &c[5*ldc], c5057.v);
  _mm256_store_ps( &c[6*ldc], c6067.v);
  _mm256_store_ps( &c[7*ldc], c7077.v);
}


inline void pack_a(int k, const float* a, int lda, float* to) {
  for(int j=0; j<k; j++) {
    const float *a_ij_ptr = &a[(j*lda)+0]; 
    *to = *a_ij_ptr;
    *(to+1) = *(a_ij_ptr+1);
    *(to+2) = *(a_ij_ptr+2);
    *(to+3) = *(a_ij_ptr+3);
    *(to+4) = *(a_ij_ptr+4);
    *(to+5) = *(a_ij_ptr+5);
    *(to+6) = *(a_ij_ptr+6);
    *(to+7) = *(a_ij_ptr+7);
    to += 8;
  }
}

inline void pack_b(int k, const float* b, int ldb, float* to) {
  int i;
  const float *b_i0_ptr = &b[0], *b_i1_ptr = &b[(1*ldb)],
              *b_i2_ptr = &b[(2*ldb)], *b_i3_ptr = &b[(3*ldb)],
              *b_i4_ptr = &b[(4*ldb)], *b_i5_ptr = &b[(5*ldb)],
              *b_i6_ptr = &b[(6*ldb)], *b_i7_ptr = &b[(7*ldb)];
  for(i=0; i<k; i++) {
    *to     = *b_i0_ptr;
    *(to+1) = *(b_i1_ptr);
    *(to+2) = *(b_i2_ptr);
    *(to+3) = *(b_i3_ptr);
    *(to+4) = *(b_i4_ptr);
    *(to+5) = *(b_i5_ptr);
    *(to+6) = *(b_i6_ptr);
    *(to+7) = *(b_i7_ptr);
    to += 8;
    b_i0_ptr++; b_i1_ptr++; b_i2_ptr++;
    b_i3_ptr++; b_i4_ptr++; b_i5_ptr++;
    b_i6_ptr++; b_i7_ptr++;
  }
}

template<int mb=128, int kb=128, int th=1>
void _m256_gemm(float* a, float* b, float* c, int m, int n, int k) {
  #pragma omp parallel for shared(a, b, c, m, n, k) default(none) collapse(1) num_threads(th)
  for(int i=0; i<k; i+=kb) {
    int ib = std::min(k-i, kb);
    float* pb = new alignas(32) float[ib*n];
    for(int ii=0; ii<m; ii+=mb) {
      int iib = std::min(m-ii, mb);
      float * pa = new alignas(32) float[ib*iib];
      float* wa = &a[i*k+ii];
      float* wb = &b[i];
      for(int iii=0; iii<n; iii+=8) {
        if(ii==0) pack_b(ib, &wb[iii*n], n, &pb[iii*ib]);
        for(int iiii=0; iiii<iib; iii+=8) {
          if(iii==0) pack_a(ib, &wa[iiii], k ,&pa[iiii*ib]);
          _8x8_m256_gemm(iib, &pa[iiii*ib], &pb[iii*ib], &c[ii+iii*n+iiii], n);
        }
      }
    }
  }
}


#endif // AVX and FMA
