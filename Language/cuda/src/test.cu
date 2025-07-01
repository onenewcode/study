#include <cstdio>
#include <cuda/std/cstdint>
#include <limits>
#include <math_constants.h>

template <typename T>
struct kv_cache {
    T *k;
    T *v;
    __device__ kv_cache(T *k_, T *v_) : k(k_), v(v_) {}
};

template <typename T>
__device__ kv_cache<T> locate_cache(
    T *const *pages,
    int64_t sbuf,      // sequence stride
    int64_t skv,       //   k to v stride
    int64_t sh,        //  kv head stride
    uint64_t const bs, // context tile
    uint64_t const head,
    uint64_t const pos) {
    sh *= head;
    sbuf *= pos % bs;
    uint8_t *page = (uint8_t *)pages[pos / bs];
    return kv_cache{
        (T *)(page + sh + sbuf),
        (T *)(page + sh + sbuf + skv),
    };
}

template <typename T>
__device__ void __attn(
    T *const *kv_pages,
    T const *q_,           // n@ x d
    T *o_,                 // n@ x d
    bool const *mask_,     // n x s
    T *m,                  // s
    T *l,                  // s
    uint64_t const n,      // sequence length
    uint64_t const d,      // head dim
    uint64_t const ts,     // = s/bs
    uint64_t const bs,     // context tile
    int64_t const sq,      //        q stride
    int64_t const so,      //        o stride
    int64_t const kv_sbuf, // sequence stride
    int64_t const kv_skv,  //   k to v stride
    int64_t const kv_sh,   //  kv head stride
    float const scale) {
    // (batch x head) x (bn)
    uint64_t const head = blockIdx.x;
    uint64_t const bn = blockDim.x;
    uint64_t const it = threadIdx.x;
    uint64_t const tn = (n + bn - 1) / bn;

    extern __shared__ T sram[];
    int tile_size = bs * d;
    T *qi = sram;
    T *kj = &sram[tile_size];
    T *vj = &sram[tile_size * 2];
    T *x = new T[bs];
    // kv
    for (uint64_t ikvb = 0; ikvb < ts; ++ikvb) {
        // 加载kv
        { // 每个线程拷贝 k/v 的一行，拷贝整个 kv block 到 local memory
            uint64_t const end = (ikvb + 1) * bs;
            for (uint64_t ikv = ikvb * bs + it, i = it; ikv < end; ikv += bn, i += bn) {
                kv_cache const cache = locate_cache(kv_pages, kv_sbuf, kv_skv, kv_sh, bs, head, ikv);
                for (uint64_t j = 0; j < d; ++j) {
                    kj[i * d + j] = cache.k[j];
                    vj[i * d + j] = cache.v[j];
                }
            }
            __syncthreads();
        }
        { // 每个线程计算 q 的一行

            for (uint64_t iqb = 0; iqb < tn; ++iqb) {
                uint64_t iq = iqb * bn + it;
                if (iq >= n) {
                    break;
                }
                // locate data 加载q
                T const *q = q_ + iq * sq;
                T *o = o_ + iq * so;
                bool const *mask = mask_ + iq * bs * ts + ikvb * bs;
                // load data
                for (uint64_t i = 0; i < d; ++i) {
                    qi[i] = q[i];
                }

                T const mi_1 = m[iq];
                T const di_1 = l[iq];

                // score = q @ k^T / √d
                T mi = mi_1;
                for (uint64_t i = 0; i < bs; ++i) {
                    if (!mask[i]) {
                        x[i] = -CUDART_INF_F;
                    } else {
                        T const *k = kj + i * d;
                        // 初始化
                        x[i] = 0;
                        for (uint64_t j = 0; j < d; ++j) {
                            x[i] += qi[j] * k[j];
                        }
                        x[i] *= scale;

                        if (x[i] > mi) {
                            mi = x[i];
                        }
                    }
                }
                // P = exp(S - row_m), row_l = rowsum(P)
                T sum = 0.;
                for (uint64_t i = 0; i < bs; ++i) {
                    x[i] = ::exp(x[i] - mi);
                    sum += x[i];
                }

                T exp = di_1 * ::exp(mi_1 - mi);
                T di = exp + sum;
                // 更新mi,di
                m[iq] = mi;
                l[iq] = di;

                T rdi = 1 / di;
                exp *= rdi;
                for (uint64_t i = 0; i < bs; ++i) {
                    x[i] *= rdi;
                }
                T *xv = new T[d];
                for (uint64_t i = 0; i < d; ++i) {
                    xv[i] = 0;
                }
                for (uint64_t i = 0; i < bs; ++i) {
                    T xi = x[i];
                    T const *v= vj + i * d;
                    for (uint64_t j = 0; j < d; ++j) {
                        xv[j] += xi * vji[j];
                    }
                }
                for (uint64_t j = 0; j < d; ++j) {
                    o[j] = o[j] * exp + xv[j];
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
    uint64_t const n,
    uint64_t const d,
    uint64_t const ts,
    uint64_t const bs,
    int64_t const sq,
    int64_t const so,
    int64_t const kv_sbuf,
    int64_t const kv_skv,
    int64_t const kv_sh,
    float const scale) {
    // 打印所有参数
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("kv_pages: %p\n", kv_pages);
        for (uint64_t i = 0; i < ts; ++i) {
            printf("  kv_pages[%lu]: %p\n", i, kv_pages[i]);
            double *page = kv_pages[i];
            for (uint64_t j = 0; j < bs * d * 2; ++j) {
                printf("    kv_pages[%lu][%lu]=%f\n", i, j, page[j]);
            }
        }
        printf("q_: %p\n", q_);
        for (uint64_t i = 0; i < n * d; ++i) {
            printf("  q_[%lu]=%f\n", i, q_[i]);
        }
        printf("o_: %p\n", o_);
        for (uint64_t i = 0; i < n * d; ++i) {
            printf("  o_[%lu]=%f\n", i, o_[i]);
        }
        printf("mask_: %p\n", (void *)mask_);
        for (uint64_t i = 0; i < n * n; ++i) {
            printf("  mask_[%lu] = %s\n", i, mask_[i] ? "true" : "false");
        }
        printf("m: %p\n", m);
        for (uint64_t i = 0; i < n; ++i) {
            printf("  m[%lu]=%f\n", i, m[i]);
        }
        printf("l: %p\n", l);
        for (uint64_t i = 0; i < n; ++i) {
            printf("  l[%lu]=%f\n", i, l[i]);
        }
        printf("n: %lu\n", n);
        printf("d: %lu\n", d);
        printf("ts: %lu\n", ts);
        printf("bs: %lu\n", bs);
        printf("sq: %ld\n", sq);
        printf("so: %ld\n", so);
        printf("kv_sbuf: %ld\n", kv_sbuf);
        printf("kv_skv: %ld\n", kv_skv);
        printf("kv_sh: %ld\n", kv_sh);
        printf("scale: %f\n", scale);
    }
    // 调用模板实现
    __attn<double>(kv_pages, q_, o_, mask_, m, l, n, d, ts, bs, sq, so, kv_sbuf, kv_skv, kv_sh, scale);
}
// 简化版：原地重排 h_kv，a、b等长，交替取 d 个元素
void interleave_h_kv_inplace(double *h_kv, size_t len, size_t d) {
    double *tmp = new double[len];
    size_t half = len / 2;
    size_t a_pos = 0, b_pos = 0, out_pos = 0;
    while (a_pos < half) {
        for (size_t i = 0; i < d; ++i) {
            tmp[out_pos++] = h_kv[a_pos++];
        }
        for (size_t i = 0; i < d; ++i) {
            tmp[out_pos++] = h_kv[half + b_pos++];
        }
    }
    for (size_t i = 0; i < len; ++i) {
        h_kv[i] = tmp[i];
    }
    delete[] tmp;
}

int main() {
    // 测试 __attn_f64 kernel（数据量加倍）
    constexpr uint64_t n = 4, s = 4, d = 4, ts = 2, bs = 2; // n, ts, h_kv等均加倍
    constexpr int64_t sq = d, so = d, kv_sbuf = d * 2 * 8, kv_skv = d * 8, kv_sh = d * 8;
    float scale = 1.0f / sqrtf((float)d);

    // 分配 host 内存
    double h_kv[bs * d * 2 * ts] = {
        0.5,
        2.3,
        -3.8,
        4.1,
        5.6,
        -6.9,
        7.2,
        8.0,
        9.4,
        -10.3,
        11.7,
        12.1,
        -13.5,
        14.8,
        15.9,
        -16.0,
        17.2,
        18.9,
        -19.5,
        20.3,
        21.0,
        -22.7,
        23.4,
        24.1,
        25.8,
        -26.6,
        27.3,
        28.9,
        -29.2,
        30.7,
        31.5,
        -32.0,
    };
    interleave_h_kv_inplace(h_kv, sizeof(h_kv) / sizeof(double), d);
    double *h_kv_pages[ts];
    for (int i = 0; i < ts; ++i) {
        h_kv_pages[i] = h_kv + i * bs * d * 2;
    }
    double h_q[n * d] = {
        1.3,
        -2.7,
        3.1,
        4.9,
        -5.5,
        6.2,
        -7.0,
        8.4,
        9.8,
        10.1,
        -11.3,
        12.6,
        13.9,
        -14.2,
        15.0,
        16.7,
    };
    double h_o[n * d] = {0};
    bool h_mask[n * n];
    for (uint64_t i = 0; i < n * s; ++i) {
        h_mask[i] = (i % n <= i / n);
    }
    double h_m[n];
    std::fill(h_m, h_m + n, std::numeric_limits<double>::lowest());
    double h_l[n] = {0};

    // 分配 device 内存
    double **d_kv_pages;
    double *d_kv;
    double *d_q, *d_o, *d_m, *d_l;
    bool *d_mask;
    cudaMalloc(&d_kv, sizeof(h_kv));
    cudaMemcpy(d_kv, h_kv, sizeof(h_kv), cudaMemcpyHostToDevice);
    cudaMalloc(&d_kv_pages, sizeof(double *) * ts);
    double *h_kv_dev_ptrs[ts];
    for (int i = 0; i < ts; ++i) {
        h_kv_dev_ptrs[i] = d_kv + i * bs * d * 2;
    }
    cudaMemcpy(d_kv_pages, h_kv_dev_ptrs, sizeof(double *) * ts, cudaMemcpyHostToDevice);
    cudaMalloc(&d_q, sizeof(h_q));
    cudaMemcpy(d_q, h_q, sizeof(h_q), cudaMemcpyHostToDevice);
    cudaMalloc(&d_o, sizeof(h_o));
    cudaMemset(d_o, 0, sizeof(h_o));
    cudaMalloc(&d_mask, sizeof(h_mask));
    cudaMemcpy(d_mask, h_mask, sizeof(h_mask), cudaMemcpyHostToDevice);
    cudaMalloc(&d_m, sizeof(h_m));
    cudaMemcpy(d_m, h_m, sizeof(h_m), cudaMemcpyHostToDevice);
    cudaMalloc(&d_l, sizeof(h_l));
    cudaMemcpy(d_l, h_l, sizeof(h_l), cudaMemcpyHostToDevice);

    // 启动 kernel
    int block = 1, thread = 2;
    size_t shared_mem = bs * d * 4 * sizeof(double);
    __attn_f64<<<block, thread, shared_mem>>>(d_kv_pages, d_q, d_o, d_mask, d_m, d_l, n, d, ts, bs, sq, so, kv_sbuf, kv_skv, kv_sh, scale);
    cudaDeviceSynchronize();

    // 拷贝输出回 host
    cudaMemcpy(h_o, d_o, sizeof(h_o), cudaMemcpyDeviceToHost);
    printf("output:\n");
    for (int i = 0; i < n * d; ++i) {
        printf("%lf ", h_o[i]);
    }
    printf("\n");

    // 释放 device 内存
    cudaFree(d_kv);
    cudaFree(d_kv_pages);
    cudaFree(d_q);
    cudaFree(d_o);
    cudaFree(d_mask);
    cudaFree(d_m);
    cudaFree(d_l);
}
