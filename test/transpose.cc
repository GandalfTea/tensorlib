
#include <iostream>
#include <chrono>
#include <immintrin.h>
#include <iomanip>

#define N 1024 

/*
inline void two_loops_pack(const int lda, const int ldb, const float* src, float* to) {
  for(int i=0; i<lda; i++) {
    for(int j=0; j<ldb; j++) {
      to[j*i] = from[j*lda+i];
    }
  }
} 

inline void one_loop_pack(const int lda, const int ldb, const float* src, float* to) {
  for(int n=0; n<lda*ldb; n++) {
    int i = n/lda; 
    int j = n%lda;
    to[n] = from[ldb*j+i];
  } 
}

inline void sse_8x4_pack(const int lda, const int ldb, const float* src, float* to) {
  __m256 r0 = _mm256_load_ps(&a[0*lda]);
  __m256 r1 = _mm256_load_ps(&a[1*lda]);
  __m256 r2 = _mm256_load_ps(&a[2*lda]);
  __m256 r3 = _mm256_load_ps(&a[3*lda]);
  __
}

inline void sse_pack(const int lda, const int ldb, const float* src, float* to) {
  
}
*/

// transpose source : https://stackoverflow.com/questions/25622745/transpose-an-8x8-float-using-avx-avx2/25627536#25627536

typedef union {
  __m256 v;
  float f[8];
} fm256;

inline void t_8x8_ps(__m256 &r0, __m256 &r1, __m256 &r2, __m256 &r3, __m256 &r4, __m256 &r5, __m256 &r6, __m256 &r7) {
  __m256 t0, t1, t2, t3, t4,t5, t6, t7,
         tt0, tt1, tt2, tt3, tt4, tt5, tt6, tt7;

 t0 = _mm256_unpacklo_ps(r0, r1);
 t1 = _mm256_unpackhi_ps(r0, r1);
 t2 = _mm256_unpacklo_ps(r2, r3);
 t3 = _mm256_unpackhi_ps(r2, r3);
 t4 = _mm256_unpacklo_ps(r4, r5);
 t5 = _mm256_unpackhi_ps(r4, r5);
 t6 = _mm256_unpacklo_ps(r6, r7);
 t7 = _mm256_unpackhi_ps(r6, r7);

 tt0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1,0,1,0));
 tt1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3,2,3,2));
 tt2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1,0,1,0));
 tt3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3,2,3,2));
 tt4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1,0,1,0));
 tt5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3,2,3,2));
 tt6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1,0,1,0));
 tt7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3,2,3,2));

 r0 = _mm256_permute2f128_ps(tt0, tt4, 0x20);
 r1 = _mm256_permute2f128_ps(tt1, tt5, 0x20);
 r2 = _mm256_permute2f128_ps(tt2, tt6, 0x20);
 r3 = _mm256_permute2f128_ps(tt3, tt7, 0x20);
 r4 = _mm256_permute2f128_ps(tt0, tt4, 0x31);
 r5 = _mm256_permute2f128_ps(tt1, tt5, 0x31);
 r6 = _mm256_permute2f128_ps(tt2, tt6, 0x31);
 r7 = _mm256_permute2f128_ps(tt3, tt7, 0x31);
}

inline void t_load_8x8_ps(const float* from, float* to, int lda, int ldb) {
  __m256 t0, t1, t2, t3, t4,t5, t6, t7,
         r0, r1, r2, r3, r4, r5, r6, r7;

  r0 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&from[0*lda+0])), _mm_load_ps(&from[4*lda+0]), 1);
  r1 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&from[1*lda+0])), _mm_load_ps(&from[5*lda+0]), 1);
  r2 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&from[2*lda+0])), _mm_load_ps(&from[6*lda+0]), 1);
  r3 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&from[3*lda+0])), _mm_load_ps(&from[7*lda+0]), 1);
  r4 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&from[0*lda+4])), _mm_load_ps(&from[4*lda+4]), 1);
  r5 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&from[1*lda+4])), _mm_load_ps(&from[5*lda+4]), 1);
  r6 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&from[2*lda+4])), _mm_load_ps(&from[6*lda+4]), 1);
  r7 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&from[3*lda+4])), _mm_load_ps(&from[7*lda+4]), 1);

  t0 = _mm256_unpacklo_ps(r0, r1);
  t1 = _mm256_unpackhi_ps(r0, r1);
  t2 = _mm256_unpacklo_ps(r2, r3);
  t3 = _mm256_unpackhi_ps(r2, r3);
  t4 = _mm256_unpacklo_ps(r4, r5);
  t5 = _mm256_unpackhi_ps(r4, r5);
  t6 = _mm256_unpacklo_ps(r6, r7);
  t7 = _mm256_unpackhi_ps(r6, r7);

  r0 = _mm256_shuffle_ps(t0, t2, 0x44);
  r1 = _mm256_shuffle_ps(t0, t2, 0xee);
  r2 = _mm256_shuffle_ps(t1, t3, 0x44);
  r3 = _mm256_shuffle_ps(t1, t3, 0xee);
  r4 = _mm256_shuffle_ps(t4, t6, 0x44);
  r5 = _mm256_shuffle_ps(t4, t6, 0xee);
  r6 = _mm256_shuffle_ps(t5, t7, 0x44);
  r7 = _mm256_shuffle_ps(t5, t7, 0xee);

  _mm256_store_ps( &to[0*ldb], r0);
  _mm256_store_ps( &to[1*ldb], r1);
  _mm256_store_ps( &to[2*ldb], r2);
  _mm256_store_ps( &to[3*ldb], r3);
  _mm256_store_ps( &to[4*ldb], r4);
  _mm256_store_ps( &to[5*ldb], r5);
  _mm256_store_ps( &to[6*ldb], r6);
  _mm256_store_ps( &to[7*ldb], r7);
}

inline void _t_ps_inner(float* from, float* to, int lda, int ldb) {
  __m256 r0, r1, r2, r3, r4, r5, r6, r7;
  r0 = _mm256_load_ps(&from[0*lda]);
  r1 = _mm256_load_ps(&from[1*lda]);
  r2 = _mm256_load_ps(&from[2*lda]);
  r3 = _mm256_load_ps(&from[3*lda]);
  r4 = _mm256_load_ps(&from[4*lda]);
  r5 = _mm256_load_ps(&from[5*lda]);
  r6 = _mm256_load_ps(&from[6*lda]);
  r7 = _mm256_load_ps(&from[7*lda]);
  t_8x8_ps(r0, r1, r2, r3, r4, r5, r6, r7);
  _mm256_store_ps( &to[0*ldb], r0);
  _mm256_store_ps( &to[1*ldb], r1);
  _mm256_store_ps( &to[2*ldb], r2);
  _mm256_store_ps( &to[3*ldb], r3);
  _mm256_store_ps( &to[4*ldb], r4);
  _mm256_store_ps( &to[5*ldb], r5);
  _mm256_store_ps( &to[6*ldb], r6);
  _mm256_store_ps( &to[7*ldb], r7);
}

template<int block, int n, int m>
inline void t_ps_load128(float* a, float* b, int lda, int ldb) {
  auto start = std::chrono::high_resolution_clock::now();
  #pragma omp parallel for shared(a, b, lda, ldb) default(none) collapse(2) num_threads(24)
  for(int i=0; i<n; i+=block) {
    for(int j=0; j<m; j+=block) {
      int mk = std::min(i+block, n);
      int ml = std::min(j+block, m);
      for(int k=i; k<mk; k+=8) {
        for(int l=j; l<ml; l+=8) {
          //_t_ps_inner(&a[k*lda+l], &b[l*ldb+k], lda, ldb);
          t_load_8x8_ps(&a[k*lda+l], &b[l*ldb+k], lda, ldb);
        }
      }
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> ms_double = end - start;
  std::cout << "transpose runtime: " << ms_double.count() << " ms " << std::endl; 
}


template<int block, int n, int m>
inline void t_ps(float* a, float* b, int lda, int ldb) {
  auto start = std::chrono::high_resolution_clock::now();
  #pragma omp parallel for shared(a, b, lda, ldb) default(none) collapse(2) num_threads(24)
  for(int i=0; i<n; i+=block) {
    for(int j=0; j<m; j+=block) {
      int mk = std::min(i+block, n);
      int ml = std::min(j+block, m);
      for(int k=i; k<mk; k+=8) {
        for(int l=j; l<ml; l+=8) {
          _t_ps_inner(&a[k*lda+l], &b[l*ldb+k], lda, ldb);
        }
      }
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> ms_double = end - start;
  std::cout << "transpose runtime: " << ms_double.count() << " ms " << std::endl; 
}

inline void t_4x4_sse(float* a, float* b, int lda, int ldb) {
  __m128 r0 = _mm_load_ps(&a[0*lda]);
  __m128 r1 = _mm_load_ps(&a[1*lda]);
  __m128 r2 = _mm_load_ps(&a[2*lda]);
  __m128 r3 = _mm_load_ps(&a[3*lda]);
  _MM_TRANSPOSE4_PS(r0, r1, r2, r3);
  _mm_store_ps(&b[0*ldb], r0);
  _mm_store_ps(&b[1*ldb], r1);
  _mm_store_ps(&b[2*ldb], r2);
  _mm_store_ps(&b[3*ldb], r3);
}

template <int block, int n, int m>
inline void t_ps_4x4_sse(float* a, float* b, int lda, int ldb) {
  auto start = std::chrono::high_resolution_clock::now();
  #pragma omp parallel for shared(a, b, lda, ldb) default(none) collapse(2) num_threads(24)
  for(int i=0; i<n; i+=block) {
    for(int j=0; j<m; j+=block) {
      int mk = std::min(i+block, n);
      int ml = std::min(j+block, m);
      for(int k=i; k<mk; k+=4) {
        for(int l=j; l<ml; l+=4) {
          t_4x4_sse(&a[k*lda+l], &b[l*ldb+k], lda, ldb);
        }
      }
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> ms_double = end - start;
  std::cout << "transpose runtime: " << ms_double.count() << " ms " << std::endl; 
}




int main() {
  float* from = new alignas(32) float[N*N];
  float* to = new alignas(32) float[N*N];
  float* a = new alignas(32) float[N*N];
  float* b = new alignas(32) float[N*N];
  float* c = new alignas(32) float[N*N];
  float* d = new alignas(32) float[N*N];
  //alignas(32) float from[N*N];
  //alignas(32) float to[N*N];
  for(int i=0; i<N*N; i++) from[i] = (float)i;
  for(int i=0; i<N*N; i++) a[i] = (float)i;
  for(int i=0; i<N*N; i++) c[i] = (float)i;

/*
  for(int i=0; i<N*N; i++) {
    if(i%N==0) std::cout << "\n";
    std::cout << std::setw(4) << from[i] << ", ";
  }
  */

  t_ps<128, N, N>(from, to, N, N);
  t_ps_load128<128, N, N>(a, b, N, N);
  //t_ps_4x4_sse<16, N, N> (c, d, N, N);
  

  /*
  int lda=N, ldb=N;
  fm256 t0, t1, t2, t3, t4,t5, t6, t7,
         tt0, tt1, tt2, tt3, tt4, tt5, tt6, tt7;
  fm256 r0, r1, r2, r3, r4, r5, r6, r7;
  r0.v = _mm256_load_ps(&from[0*lda]);
  r1.v = _mm256_load_ps(&from[1*lda]);
  r2.v = _mm256_load_ps(&from[2*lda]);
  r3.v = _mm256_load_ps(&from[3*lda]);
  r4.v = _mm256_load_ps(&from[4*lda]);
  r5.v = _mm256_load_ps(&from[5*lda]);
  r6.v = _mm256_load_ps(&from[6*lda]);
  r7.v = _mm256_load_ps(&from[7*lda]);

std::cout << std::endl;
std::cout << "_mm256_load_ps: " << std::endl;
std::cout << "r0: ";
for(int i=0; i<8; i++) std::cout << std::setw(4) << r0.f[i]    << ", ";
std::cout << std::endl;
std::cout << "r1: ";
for(int i=0; i<8; i++) std::cout << std::setw(4) << r1.f[i]    << ", ";
std::cout << std::endl;
std::cout << "r2: ";
for(int i=0; i<8; i++) std::cout << std::setw(4) << r2.f[i]    << ", ";
std::cout << std::endl;
std::cout << "r3: ";
for(int i=0; i<8; i++) std::cout << std::setw(4) << r3.f[i]    << ", ";
std::cout << std::endl;
std::cout << "r4: ";
for(int i=0; i<8; i++) std::cout << std::setw(4) << r4.f[i]    << ", ";
std::cout << std::endl;
std::cout << "r5: ";
for(int i=0; i<8; i++) std::cout << std::setw(4) << r5.f[i]    << ", ";
std::cout << std::endl;
std::cout << "r6: ";
for(int i=0; i<8; i++) std::cout << std::setw(4) << r6.f[i]    << ", ";
std::cout << std::endl;
std::cout << "r7: ";
for(int i=0; i<8; i++) std::cout << std::setw(4) << r7.f[i]    << ", ";
std::cout << std::endl;


 t0.v = _mm256_unpacklo_ps(r0.v, r1.v);
 t2.v = _mm256_unpacklo_ps(r2.v, r3.v);
 t4.v = _mm256_unpacklo_ps(r4.v, r5.v);
 t6.v = _mm256_unpacklo_ps(r6.v, r7.v);

 t1.v = _mm256_unpackhi_ps(r0.v, r1.v);
 t3.v = _mm256_unpackhi_ps(r2.v, r3.v);
 t5.v = _mm256_unpackhi_ps(r4.v, r5.v);
 t7.v = _mm256_unpackhi_ps(r6.v, r7.v);

std::cout << std::endl;
std::cout << "_mm256_unpacklo_ps: " << std::endl;
std::cout << "t0 (r0, r1): ";
for(int i=0; i<8; i++) std::cout << std::setw(4) << t0.f[i]   << ", ";
std::cout << std::endl;
std::cout << "t1 (r0, r1): ";
for(int i=0; i<8; i++) std::cout << std::setw(4) << t1.f[i]  << ", ";
std::cout << std::endl;
std::cout << "t2 (r2, r3): ";
for(int i=0; i<8; i++) std::cout << std::setw(4) << t2.f[i]   << ", ";
std::cout << std::endl;
std::cout << "t3 (r2, r3): ";
for(int i=0; i<8; i++) std::cout << std::setw(4) << t3.f[i]   << ", ";
std::cout << std::endl;
std::cout << "t4 (r4, r5): ";
for(int i=0; i<8; i++) std::cout << std::setw(4) << t4.f[i]   << ", ";
std::cout << std::endl;
std::cout << "t5 (r4, r5): ";
for(int i=0; i<8; i++) std::cout << std::setw(4) << t5.f[i]  << ", ";
std::cout << std::endl;
std::cout << "t6 (r6, r7): ";
for(int i=0; i<8; i++) std::cout << std::setw(4) << t6.f[i]   << ", ";
std::cout << std::endl;
std::cout << "t7 (r6, r7): ";
for(int i=0; i<8; i++) std::cout << std::setw(4) << t7.f[i]   << ", ";
std::cout << std::endl;


 tt0.v = _mm256_shuffle_ps(t0.v, t2.v, _MM_SHUFFLE(1,0,1,0));
 tt2.v = _mm256_shuffle_ps(t1.v, t3.v, _MM_SHUFFLE(1,0,1,0));
 tt4.v = _mm256_shuffle_ps(t4.v, t6.v, _MM_SHUFFLE(1,0,1,0));
 tt6.v = _mm256_shuffle_ps(t5.v, t7.v, _MM_SHUFFLE(1,0,1,0));

 tt1.v = _mm256_shuffle_ps(t0.v, t2.v, _MM_SHUFFLE(3,2,3,2));
 tt3.v = _mm256_shuffle_ps(t1.v, t3.v, _MM_SHUFFLE(3,2,3,2));
 tt5.v = _mm256_shuffle_ps(t4.v, t6.v, _MM_SHUFFLE(3,2,3,2));
 tt7.v = _mm256_shuffle_ps(t5.v, t7.v, _MM_SHUFFLE(3,2,3,2));

std::cout << std::endl;
std::cout << "_mm256_shuffle_ps : " << std::endl;
std::cout << "tt0 (t0, t2): ";
for(int i=0; i<8; i++) std::cout << std::setw(4) << tt0.f[i]  << ", ";
std::cout << std::endl;
std::cout << "tt1 (t0, t2): ";
for(int i=0; i<8; i++) std::cout << std::setw(4) << tt1.f[i]  << ", ";
std::cout << std::endl;
std::cout << "tt2 (t1, t3): ";
for(int i=0; i<8; i++) std::cout << std::setw(4) << tt2.f[i]  << ", ";
std::cout << std::endl;
std::cout << "tt3 (t1, t3): ";
for(int i=0; i<8; i++) std::cout << std::setw(4) << tt3.f[i] << ", ";
std::cout << std::endl;
std::cout << "tt4 (t4, t6): ";
for(int i=0; i<8; i++) std::cout<< std::setw(4) << tt4.f[i] << ", ";
std::cout << std::endl;
std::cout << "tt5 (t4, t6): ";
for(int i=0; i<8; i++) std::cout << std::setw(4) << tt5.f[i]  << ", ";
std::cout << std::endl;
std::cout << "tt6 (t5, t7): ";
for(int i=0; i<8; i++) std::cout << std::setw(4) << tt6.f[i]  << ", ";
std::cout << std::endl;
std::cout << "tt7 (t5, t7): ";
for(int i=0; i<8; i++) std::cout << tt7.f[i]  << ", ";
std::cout << std::endl;

 r0.v = _mm256_permute2f128_ps(tt0.v, tt4.v, 0x20);
 r1.v = _mm256_permute2f128_ps(tt1.v, tt5.v, 0x20);
 r2.v = _mm256_permute2f128_ps(tt2.v, tt6.v, 0x20);
 r3.v = _mm256_permute2f128_ps(tt3.v, tt7.v, 0x20);

 r4.v = _mm256_permute2f128_ps(tt0.v, tt4.v, 0x31);
 r5.v = _mm256_permute2f128_ps(tt1.v, tt5.v, 0x31);
 r6.v = _mm256_permute2f128_ps(tt2.v, tt6.v, 0x31);
 r7.v = _mm256_permute2f128_ps(tt3.v, tt7.v, 0x31);

std::cout << std::endl;
std::cout << "_mm256_permute2f128_ps 0x20 " << std::endl;
std::cout << "r0 (tt0, tt4): ";
for(int i=0; i<8; i++) std::cout << std::setw(4) << r0.f[i] << ", ";
std::cout << std::endl;
std::cout << "r1 (tt1, tt5): ";
for(int i=0; i<8; i++) std::cout << std::setw(4) << r1.f[i] << ", ";
std::cout << std::endl;
std::cout << "r2 (tt2, tt6): ";
for(int i=0; i<8; i++) std::cout << std::setw(4) << r2.f[i] << ", ";
std::cout << std::endl;
std::cout << "r3 (tt3, tt7): ";
for(int i=0; i<8; i++) std::cout << std::setw(4) << r3.f[i] << ", ";
std::cout << std::endl;

std::cout << std::endl;
std::cout << "_mm256_permute2f128_ps 0x32 " << std::endl;
std::cout << "r4 (tt0, tt4): ";
for(int i=0; i<8; i++) std::cout << std::setw(4) << r4.f[i]  << ", ";
std::cout << std::endl;
std::cout << "r5 (tt1, tt5): ";
for(int i=0; i<8; i++) std::cout << std::setw(4) << r5.f[i]  << ", ";
std::cout << std::endl;
std::cout << "r6 (tt2, tt6): ";
for(int i=0; i<8; i++) std::cout << std::setw(4) << r6.f[i] << ", ";
std::cout << std::endl;
std::cout << "r7 (tt3, tt7): ";
for(int i=0; i<8; i++) std::cout << std::setw(4) << r7.f[i]  << ", ";
std::cout << std::endl;
*/
  /*
  auto start = std::chrono::high_resolution_clock::now();
  for(int i=0; i<N; i+=8) {
    for(int j=0; j<N; j+=8) {
      std::cout << i <<" " << j << std::endl;
      _t_ps_inner(&a[i*N+j], &b[j*N+i], N, N);
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> ms_double = end - start;
  std::cout << "\n\n" << "transpose runtime: " << ms_double.count() << " ms "; 
  */

/*
  std::cout << "\n\n";
  for(int i=0; i<N*N; i++) {
    if(i%N==0) std::cout << "\n";
    std::cout << std::setw(3) << to[i] << ", ";
  }
  std::cout << "\n\n";
  */
}
