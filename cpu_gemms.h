

// sorry MSVC bros
// microsoft is baka
#if defined(__AVX__) && (defined(__FMA__) || defined(__FMA4__)) 

#include <memory>
#include <stdint.h>
#include <omp.h>
#include <immintrin.h>


// bit fields

// 0x01 - extended basic information

struct eax1_intel {
	unsigned stepping_id: 4;
	unsigned model_id: 4;
	unsigned family_id: 4;
	unsigned processor_type: 2;
	unsigned : 2; 
	unsigned extended_model: 4;
	unsigned extended_family: 8;
	unsigned : 4;
};

struct eax1_amd {
	unsigned stepping_id: 4;
	unsigned model_id: 4;
	unsigned family_id: 4;
	unsigned : 4; 
	unsigned extended_model: 4;
	unsigned extended_family: 8;
	unsigned : 4;
};

typedef union {
  eax1_intel intel;
  eax1_amd amd;
  uint32_t eax;
} eax1b;

struct ebx1 {
	unsigned brand_index: 8;
	unsigned CGLUSH_line_size: 8;
	unsigned logical_processors: 8;
	unsigned initial_APICID: 8;
};

typedef union {
  ebx1 bits;
  uint32_t ebx;
} ebx1b;

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

struct ecx1_amd {
  unsigned SSE3: 1;
  unsigned : 12;
  unsigned CMPXCHG16B: 1;
  unsigned : 18;
};

typedef union {
  ecx1_intel intel;
  ecx1_amd amd;
  uint32_t ecx;
} ecx1b;

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

struct edx1_amd {
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
};

typedef union {
  edx1_intel intel;
  edx1_amd amd;
  uint32_t edx;
} edx1b;



// helpers

void _cpuid(uint32_t op, uint32_t& eax, uint32_t& ebx, uint32_t& ecx, uint32_t& edx) {
  asm volatile(
    "cpuid"
    : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
    : "a"(op)
    : "cc" 
  );
}

uint32_t cpuid_highest_leaf() {
  uint32_t eax, ebx, ecx, edx;
  _cpuid(0x0, eax, ebx, ecx, edx);
  return eax;
}

// my cpu doesn't support this so it's untested
uint32_t* cpuid_cache_info() {
  uint32_t ret[4];
  ret[0] == 0x0;
  _cpuid(0x16, ret[0], ret[1], ret[2], ret[3]);
  return ret;
}

// 0x02 - cache and TLB information

uint32_t* cpuid_cache_tlb() {
  uint32_t ret[4];
  ret[0] == 0x0;
  _cpuid(0x02, ret[0], ret[1], ret[2], ret[3]);
  return ret;
}

struct {
  bool r;
  uint64_t size; // in bytes
  int8_t associative; 
  uint8_t line_size; // in bytes
} cache_data;

struct {
  uint64_t pages;
  int8_t associative;
  uint8_t entries;
} tlb_data;

struct {
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
} cpuid2_ret;

cpuid2_ret read_cache_tlb() {
  uint32_t* regs = cpuid_cache_tlb(); 
  cpuid2_ret ret = {{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0}};
  for(int i=0; i<4; i++) {
    if(i!=0) {
      if((int)(regs[i] >> 31) != 0) {
        continue; // reserved
      } else {
        for(int j=0; j<31; j++) {
          if( (int)(regs[i] >> j) == 0x00h) {
            continue; // null
          } else if ( (int)(regs[i] >> j) == 0x01h) {
            switch( (int)(regs[i] >> j) ) {
              case 0x00h: continue;
              case 0x01h: ret.ins_tlb = (tlb_data){4000, 4, 32}; continue;
              case 0x02h: ret.ins_tlb = (tlb_data){4000000, 0, 2}; continue;
              case 0x03h: ret.data_tlb0 = (tlb_data){4000, 4, 64}; continue;
              case 0x04h: ret.data_tlb0 = (tlb_data){4000000, 4, 8}; continue;
              case 0x05h: ret.data_tlb1 = (tlb_data){4000000, 4, 32}; continue;
              case 0x06h: ret.data.l1_ins = (cache_data){1, 8000, 4, 32}; continue;
              case 0x08h: ret.data.l1_ins = (cache_data){1, 16000, 4, 32}; continue;
              case 0x09h: ret.data.l1_ins = (cache_data){1, 32000, 4, 64}; continue;
              case 0x0ah: ret.data.l1_data = (cache_data){1, 8000, 2, 32}; continue;
              case 0x0bh: ret.data.ins_tlb = (tlb_data){4000, 4, 4}; continue;
              case 0x0ch: ret.data.l1_data = (cache_data){1, 16000, 4, 32}; continue;
              case 0x0dh: ret.data.l1_data = (cache_data){1, 16000, 4, 64}; continue;
              case 0x0eh: ret.data.l1_data = (cache_data){1, 24000, 6, 64}; continue;
              case 0x1dh: ret.data.l2 = (cache_data){1, 128000, 2, 64}; continue;
              case 0x21h: ret.data.l2 = (cache_data){1, 256000, 8, 64}; continue;
              case 0x22h: ret.data.l3 = (cache_data){1, 512000, 4, 64}; continue;
              case 0x23h: ret.data.l3 = (cache_data){1, 1000000, 8, 64}; continue;
              case 0x24h: ret.data.l2 = (cache_data){1, 1000000, 16, 64}; continue;
              case 0x25h: ret.data.l3 = (cache_data){1, 2000000, 8, 64}; continue;
              case 0x29h: ret.data.l3 = (cache_data){1, 4000000, 8, 64}; continue;
              case 0x2ch: ret.data.l1_data = (cache_data){1, 32000, 8, 64}; continue;
              case 0x30h: ret.data.l1_ins = (cache_data){1, 32000, 8, 64}; continue;
              case 0x40h: ret.data.l2 = (cache_data){0, 0, 0, 0}; continue;
              case 0x41h: ret.data.l2 = (cache_data){1, 128000, 4, 32}; continue;
              case 0x42h: ret.data.l2 = (cache_data){1, 256000, 4, 32}; continue;
              case 0x43h: ret.data.l2 = (cache_data){1, 512000, 4, 32}; continue;
              case 0x44h: ret.data.l2 = (cache_data){1, 1000000, 4, 32}; continue;
              case 0x45h: ret.data.l2 = (cache_data){1, 2000000, 4, 32}; continue;
              case 0x46h: ret.data.l3 = (cache_data){1, 4000000, 4, 64}; continue;
              case 0x47h: ret.data.l3 = (cache_data){1, 8000000, 8, 64}; continue;
              case 0x48h: ret.data.l2 = (cache_data){1, 3000000, 12, 64}; continue;
              case 0x49h: ret.data.l2 = (cache_data){1, 4000000, 16, 64}; continue; // For Intel Xeon family 0fh, model 06h this is l3
              case 0x4ah: ret.data.l3 = (cache_data){1, 6000000, 12, 64}; continue;
              case 0x4bh: ret.data.l3 = (cache_data){1, 8000000, 16, 64}; continue;
              case 0x4ch: ret.data.l3 = (cache_data){1, 12000000, 12, 64}; continue;
              case 0x4dh: ret.data.l3 = (cache_data){1, 16000000, 16, 64}; continue;
              case 0x4eh: ret.data.l2 = (cache_data){1, 6000000, 24, 64}; continue;
              case 0x4fh: ret.data.ins_tbl = (tbl_data){4000, -1, 32}; continue; 
              case 0x50h: ret.data.ins_tbl = (tbl_data){4000000, -1, 64}; continue; // 4kB and 2MB or 4MB
              case 0x51h: ret.data.ins_tbl = (tbl_data){4000000, -1, 128}; continue; // 4kB and 2MB or 4MB
              case 0x52h: ret.data.ins_tbl = (tbl_data){4000000, -1, 256}; continue; // 4kB and 2MB or 4MB
              case 0x55h: ret.data.ins_tbl = (tbl_data){4000000, 0, 7}; continue; // 2MB or 4MB
              case 0x56h: ret.data.data_tbl0 = (tbl_data){4000000, 4, 16}; continue; 
              case 0x57h: ret.data.data_tbl0 = (tbl_data){4000, 4, 16}; continue; 
              case 0x59h: ret.data.data_tbl0 = (tbl_data){4000, 0, 16}; continue; 
              case 0x5ah: ret.data.data_tbl0 = (tbl_data){4000000, 4, 32}; continue; // 2MB or 4MB 
              case 0x5bh: ret.data.data_tbl = (tbl_data){4000000, -1, 16}; continue; // 4MB AND 4MB pages
              case 0x5ch: ret.data.data_tbl = (tbl_data){4000000, -1, 128}; continue; // 4MB AND 4MB pages 
              case 0x5dh: ret.data.data_tbl = (tbl_data){4000000, -1, 128}; continue; // 4MB AND 4MB pages 
              case 0x60h: ret.data.l1_data = (cache_data){1, 16000, 8, 64}; continue; 
              case 0x61h: ret.data.ins_tlb = (tlb_data){4000000, 0, 48}; continue;  // 2MB or 4MB
              case 0x63h: ret.data.data_tlb = (tlb_data){4000000, 4, 32}; continue;  // 2MB or 4MB and a separate array with 2GB pages, 4, 4ent
              case 0x64h: ret.data.data_tlb = (tlb_data){4000, 4, 512}; continue; 
              case 0x66h: ret.data.l1_data = (cache_data){1, 8000, 4, 64}; continue; 
              case 0x67h: ret.data.l1_data = (cache_data){1, 16000, 4, 64}; continue; 
              case 0x68h: ret.data.l1_data = (cache_data){1, 32000, 4, 64}; continue; 
              case 0x6ah: ret.data.utlb = (tlb_data){4000, 8, 64}; continue; 
              case 0x6bh: ret.data.dtlb = (tlb_data){4000, 8, 256}; continue; 
              case 0x6ch: ret.data.dtlb = (tlb_data){4000000, 8, 128}; continue; // 2M or 4M? MB?
              case 0x6dh: ret.data.dtlb = (tlb_data){1000000000, 0, 16}; continue; 
              // skip 0x70h, 0x71h, 0x72h - trace cache
              case 0x76h: ret.data.ins_tlb = (tlb_data){4000000, 0, 8}; continue; // 2M or 4M? MB?
              case 0x78h: ret.data.l2 = (cache_data){1, 1000000, 4, 64}; continue; 
              case 0x79h: ret.data.l2 = (cache_data){1, 128000, 8, 64}; continue; 
              case 0x7ah: ret.data.l2 = (cache_data){1, 256000, 8, 64}; continue; 
              case 0x7bh: ret.data.l2 = (cache_data){1, 512000, 8, 64}; continue; 
              case 0x7ch: ret.data.l2 = (cache_data){1, 1000000, 8, 64}; continue; 
              case 0x7dh: ret.data.l2 = (cache_data){1, 2000000, 8, 64}; continue; 
              case 0x7fh: ret.data.l2 = (cache_data){1, 512000, 2, 64}; continue; 
              case 0x80h: ret.data.l2 = (cache_data){1, 512000, 8, 64}; continue; 
              case 0x82h: ret.data.l2 = (cache_data){1, 256000, 8, 32}; continue; 
              case 0x83h: ret.data.l2 = (cache_data){1, 512000, 8, 32}; continue; 
              case 0x84h: ret.data.l2 = (cache_data){1, 1000000, 8, 32}; continue; 
              case 0x85h: ret.data.l2 = (cache_data){1, 2000000, 8, 32}; continue; 
              case 0x86h: ret.data.l2 = (cache_data){1, 512000, 4, 64}; continue; 
              case 0x87h: ret.data.l2 = (cache_data){1, 1000000, 8, 64}; continue; 
              case 0xa0h: ret.data.dtlb = (tlb_data){4000, 0, 32}; continue; 
              case 0xb0h: ret.data.ins_tlb = (tlb_data){4000, 4, 128}; continue; 
              case 0xb1h: ret.data.ins_tlb = (tlb_data){2000000, 4, 8}; continue; // or 4M, 4-way, 4 entries
              case 0xb2h: ret.data.ins_tlb = (tlb_data){4000, 4, 64}; continue; 
              case 0xb3h: ret.data.data_tlb = (tlb_data){4000, 4, 128}; continue;
              case 0xb4h: ret.data.data_tlb1 = (tlb_data){4000, 4, 256}; continue;
              case 0xb5h: ret.data.ins_tlb = (tlb_data){4000, 8, 64}; continue;
              case 0xb6h: ret.data.ins_tlb = (tlb_data){4000, 8, 128}; continue;
              case 0xbah: ret.data.data_tlb1 = (tlb_data){4000, 4, 64}; continue;
              case 0xc0h: ret.data.data_tlb = (tlb_data){4000000, 4, 8}; continue; // 4kB and 4MB
              case 0xc1h: ret.data.stlb = (tlb_data){2000000, 8, 1024}; continue; // 4kB and 2MB
              case 0xc2h: ret.data.dtlb = (tlb_data){2000000, 4, 16}; continue; // 4kB and 2MB
              case 0xc3h: ret.data.stlb = (tlb_data){2000000, 6, 1536}; continue; // 4kB and 2MB also 1GB pages, 4-way, 16 entries
              case 0xc4h: ret.data.dtlb = (tlb_data){4000000, 4, 32}; continue; // 2MB and 4MB
              case 0xcah: ret.data.stlb = (tlb_data){4000, 4, 512}; continue;
              case 0xd0h: ret.data.l3 = (cache_data){1, 512000, 4, 64}; continue;
              case 0xd1h: ret.data.l3 = (cache_data){1, 1000000, 4, 64}; continue;
              case 0xd2h: ret.data.l3 = (cache_data){1, 2000000, 4, 64}; continue;
              case 0xd6h: ret.data.l3 = (cache_data){1, 1000000, 8, 64}; continue;
              case 0xd7h: ret.data.l3 = (cache_data){1, 2000000, 8, 64}; continue;
              case 0xd8h: ret.data.l3 = (cache_data){1, 4000000, 8, 64}; continue;
              case 0xdch: ret.data.l3 = (cache_data){1, 1500000, 12, 64}; continue;
              case 0xddh: ret.data.l3 = (cache_data){1, 3000000, 12, 64}; continue;
              case 0xdeh: ret.data.l3 = (cache_data){1, 6000000, 12, 64}; continue;
              case 0xe2h: ret.data.l3 = (cache_data){1, 2000000, 16, 64}; continue;
              case 0xe3h: ret.data.l3 = (cache_data){1, 4000000, 16, 64}; continue;
              case 0xe4h: ret.data.l3 = (cache_data){1, 8000000, 16, 64}; continue;
              case 0xeah: ret.data.l3 = (cache_data){1, 12000000, 24, 64}; continue;
              case 0xebh: ret.data.l3 = (cache_data){1, 18000000, 24, 64}; continue;
              case 0xech: ret.data.l3 = (cache_data){1, 24000000, 24, 64}; continue;
              case 0xfeh: ret = {{0,0x18,0,0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0}}; return; // use leaf 0x18h 
              case 0xffh: ret = {{0,0x4,0,0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0}}; return; // use leaf 0x4 
              default: ret = {{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0}}; return; // unexpected error 
            }
          }
        }
      }
    }
  }
}

#endif // AVX and FMA
