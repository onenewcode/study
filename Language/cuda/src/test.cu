#include <cuda/std/cstdint>
#include <math_constants.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

template <typename T>
struct kv_cache {
    T *k;
    T *v;
    __device__ kv_cache(T *k_, T *v_) : k(k_), v(v_) {}
};

template <typename T>
__device__ kv_cache<T> locate_cache(
    T *const *pages,
    int64_t  sbuf,      // sequence stride
    int64_t  skv,       //   k to v stride
    int64_t  sh,        //  kv head stride
    uint64_t  const bs, // context tile
    uint64_t  const head,
    uint64_t  const pos) {
    sh *= head;
    sbuf *= pos % bs;
    uint8_t *page = (uint8_t *)pages[pos / bs];
    return kv_cache<T>((T *)(page + sh + sbuf), (T *)(page + sh + sbuf + skv));
}

template <typename T>
__device__ void __attn(
    T *const *kv_pages,
    T const *q_,           // n@ x d
    T *o_,                 // n@ x d
    bool const *mask_,     // n x s
    T *m,                  // s
    T *l,                  // s
    uint64_t  const n,      // sequence length
    uint64_t  const d,      // head dim
    uint64_t  const ts,     // = s/bs
    uint64_t  const bs,     // context tile
    int64_t  const sq,      //        q stride
    int64_t  const so,      //        o stride
    int64_t  const kv_sbuf, // sequence stride
    int64_t  const kv_skv,  //   k to v stride
    int64_t  const kv_sh,   //  kv head stride
    float const scale) {
    // (batch x head) x (bn)
    uint64_t  const head = blockIdx.x;
    uint64_t  const bn = blockDim.x;
    uint64_t  const it = threadIdx.x;
    uint64_t  const tn = (n + bn - 1) / bn;

    extern __shared__ T sram[];
    int tile_size = bs * d;
    T *qi = sram;
    T *kj = &sram[tile_size];
    T *vj = &sram[tile_size * 2];
    T *x = &sram[tile_size * 3];
    // kv
    for (uint64_t  ikvb = 0; ikvb < ts; ++ikvb) {
        // 加载kv
        { // 每个线程拷贝 k/v 的一行，拷贝整个 kv block 到 local memory
            uint64_t  const end = (ikvb + 1) * bs;
            for (uint64_t  ikv = ikvb * bs + it, i = it; ikv < end; ikv += bn, i += bn) {
                kv_cache const cache = locate_cache(kv_pages, kv_sbuf, kv_skv, kv_sh, bs, head, ikv);
                for (uint64_t  j = 0; j < d; ++j) {
                    kj[i * d + j] = cache.k[j];
                    vj[i * d + j] =cache.v[j];
                }
            }
            __syncthreads();
        }
        { // 每个线程计算 q 的一行

            for (uint64_t  iqb = 0; iqb < tn; ++iqb) {
                uint64_t  iq = iqb * bn + it;
                if (iq >= n) {
                    break;
                }
                // locate data 加载q
                T const *q = q_ + iq * sq;
                T *o = o_ + iq * so;
                bool const *mask = mask_ + iq * n + ikvb * bs;
                // load data
                for (uint64_t  i = 0; i < d; ++i) {
                    qi[i] = q[i];
                }

                T const mi_1 = m[iq];
                T const di_1 = l[iq];

                // score = q @ k^T / √d
                T mi = mi_1;
                for (uint64_t  i = 0; i < bs; ++i) {
                    if (!mask[i]) {
                        x[i] = -CUDART_INF_F;
                    } else {
                        T const *k = kj + i * d;

                        for (uint64_t  j = 0; j < d; ++j) {
                            x[i] += qi[j] * kj[j];
                        }
                        x[i] *= scale;

                        if (x[i] > mi) {
                            mi = x[i];
                        }
                    }
                }
                // P = exp(S - row_m), row_l = rowsum(P)
                T sum = 0;
                for (uint64_t  i = 0; i < bs; ++i) {
                    x[i] = std::exp(x[i] - mi);
                    sum += x[i];
                }

                T exp = di_1 * std::exp(mi_1 - mi);
                T di = exp + sum;
                // 更新mi,di
                m[iq] = mi;
                l[iq] = di;

                T rdi = 1 / di;
                exp *= rdi;
                for (uint64_t  i = 0; i < bs; ++i) {
                    x[i] *= rdi;
                }
            }
            __syncthreads();
        }
    }
}
extern "C" __global__ void __attn_f64(
    double *const *kv_pages,
    double const *q_,
    double *o_,
    bool const *mask_,
    double *m,
    double *l,
    uint64_t  const n,
    uint64_t  const d,
    uint64_t  const ts,
    uint64_t  const bs,
    int64_t  const sq,
    int64_t  const so,
    int64_t  const kv_sbuf,
    int64_t  const kv_skv,
    int64_t  const kv_sh,
    float const scale) {
    // 调用模板实现
    __attn<double>(kv_pages, q_, o_, mask_, m, l, n, d, ts, bs, sq, so, kv_sbuf, kv_skv, kv_sh, scale);
}

// CPU reference implementation for masked attention (double precision)
void cpu_attention_with_mask(const double* q, const double* k, const double* v, const uint8_t* mask, double* out, int n, int d) {
    for (int i = 0; i < n; ++i) {
        std::vector<double> scores(n, 0.0);
        double max_score = -1e30;
        for (int j = 0; j < n; ++j) {
            if (!mask[i * n + j]) {
                scores[j] = -1e30;
            } else {
                double dot = 0.0;
                for (int kidx = 0; kidx < d; ++kidx) {
                    dot += q[i * d + kidx] * k[j * d + kidx];
                }
                scores[j] = dot / std::sqrt((double)d);
            }
            if (scores[j] > max_score) max_score = scores[j];
        }
        // softmax
        double sum = 0.0;
        for (int j = 0; j < n; ++j) {
            scores[j] = std::exp(scores[j] - max_score);
            sum += scores[j];
        }
        for (int kidx = 0; kidx < d; ++kidx) {
            double val = 0.0;
            for (int j = 0; j < n; ++j) {
                val += scores[j] / sum * v[j * d + kidx];
            }
            out[i * d + kidx] = val;
        }
    }
}

void test_attn_f64() {
    constexpr int n = 4, d = 3;
    std::vector<double> q(n * d), k(n * d), v(n * d);
    std::vector<double> o(n * d);
    std::fill(o.begin(), o.end(), 0.0);
    std::vector<double> o_cpu(n * d);
    std::fill(o_cpu.begin(), o_cpu.end(), 0.0);
    std::vector<uint8_t> mask(n * n);
    std::fill(mask.begin(), mask.end(), 1);
    // Set a mask: mask out the last column for the last row
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) mask[i * n + j] = 1;
    mask[(n-1) * n + (n-1)] = 0;
    // Fill q, k, v with some values
    for (int i = 0; i < n * d; ++i) {
        q[i] = 0.1 * (i + 1);
        k[i] = 0.2 * (i + 1);
        v[i] = 0.3 * (i + 1);
    }
    // CPU reference
    cpu_attention_with_mask(q.data(), k.data(), v.data(), mask.data(), o_cpu.data(), n, d);

    // Allocate device memory
    double *d_q, *d_k, *d_v, *d_o, *d_m, *d_l;
    uint8_t *d_mask;
    cudaMalloc(&d_q, n * d * sizeof(double));
    cudaMalloc(&d_k, n * d * sizeof(double));
    cudaMalloc(&d_v, n * d * sizeof(double));
    cudaMalloc(&d_o, n * d * sizeof(double));
    cudaMalloc(&d_m, n * sizeof(double));
    cudaMalloc(&d_l, n * sizeof(double));
    cudaMalloc(&d_mask, n * n * sizeof(uint8_t));
    cudaMemcpy(d_q, q.data(), n * d * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, k.data(), n * d * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v.data(), n * d * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask.data(), n * n * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemset(d_o, 0, n * d * sizeof(double));
    cudaMemset(d_m, 0, n * sizeof(double));
    cudaMemset(d_l, 0, n * sizeof(double));

    // Prepare kv_pages (single page for this test)
    double *kv_pages_host[1];
    double *kv_block;
    cudaMalloc(&kv_block, 2 * n * d * sizeof(double));
    cudaMemcpy(kv_block, k.data(), n * d * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(kv_block + n * d, v.data(), n * d * sizeof(double), cudaMemcpyHostToDevice);
    kv_pages_host[0] = kv_block;
    double **d_kv_pages;
    cudaMalloc(&d_kv_pages, sizeof(double*));
    cudaMemcpy(d_kv_pages, kv_pages_host, sizeof(double*), cudaMemcpyHostToDevice);

    // Kernel launch config
    dim3 grid(1);
    dim3 block(n);
    size_t shared_mem = 4 * n * d * sizeof(double);
    float scale = 1.0 / std::sqrt((float)d);
    __attn_f64<<<grid, block, shared_mem>>>(d_kv_pages, d_q, d_o, (bool*)d_mask, d_m, d_l, n, d, 1, n, d, d, 0, n * d, 0, scale);
    cudaDeviceSynchronize();
    cudaMemcpy(o.data(), d_o, n * d * sizeof(double), cudaMemcpyDeviceToHost);

    // Compare results
    std::cout << "CPU vs CUDA output (row major):\n";
    for (int i = 0; i < n * d; ++i) {
        std::cout << o_cpu[i] << "\t" << o[i] << "\n";
        assert(std::abs(o_cpu[i] - o[i]) < 1e-6);
    }
    std::cout << "Test passed!\n";

    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_o); cudaFree(d_m); cudaFree(d_l); cudaFree(d_mask); cudaFree(kv_block); cudaFree(d_kv_pages);
}

int main() {
    test_attn_f64();
    return 0;
}
