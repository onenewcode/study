#include <cuda/std/cstdint>
#include <math_constants.h>

template <typename T>
struct kv_cache {
    T *k;
    T *v;
    __device__ kv_cache(T *k_, T *v_) : k(k_), v(v_) {}
};

template <typename T>
__device__ void __attn(
    T *const **kv_pages,    // [batch][page]
    T *const *q_,           // batch [n@ x d]
    T **o_,                 // batch [n@ x d]
    bool *const *mask_,     // batch [n x s]
    T **m,                  // batch [s]
    T **l,                  // batch [s]
    uint64_t const *n,      // batch [sequence length]
    uint64_t const d,       // head dim
    uint64_t const *ts,     // batch [s/bs]
    uint64_t const bs,      // context tile
    uint64_t const g,       // GQA
    int64_t const sq,       // q stride
    int64_t const so,       // o stride
    int64_t const *kv_sbuf, // batch [sequence stride]
    int64_t const *kv_skv,  // batch [k to v stride]
    int64_t const *kv_sh,   // batch [kv head stride]
    float const scale) {
    // (batch x head) x (bn)
    uint64_t const b = blockIdx.y;
    uint64_t const head = blockIdx.x;
    uint64_t const bn = blockDim.x;
    uint64_t const it = threadIdx.x;
    uint64_t const tn = (n[b] + bn - 1) / bn;

    // 当前批次参数
    uint64_t const n_b = n[b];
    uint64_t const ts_b = ts[b];
    int64_t const kv_sbuf_b = kv_sbuf[b];
    int64_t const kv_skv_b = kv_skv[b];
    int64_t const kv_sh_b = kv_sh[b];

    // 目前把所有维度当作连续的
    extern __shared__ T sram[];
    T *kj = sram;
    T *vj = sram + bs * d;

    // 当前批次指针
    T const *q_batch = q_[b];
    T *o_batch = o_[b];
    bool const *mask_batch = mask_[b];
    T *m_batch = m[b];
    T *l_batch = l[b];
    T *const *pages_batch = kv_pages[b];

    // 当前头在Q/K/V中的偏移
    T const *q = q_batch + head * n_b * d;
    T *o = o_batch + head * n_b * d;

    for (uint64_t iqb = 0; iqb < tn; ++iqb) {
        uint64_t iq = iqb * bn + it;
        if (iq >= n_b) {
            continue;
        }

        // 加载当前查询位置的状态
        T mi = m_batch[head * n_b + iq];
        T li = l_batch[head * n_b + iq];
        T *oi = o + iq * d;

        for (uint64_t ikvb = 0; ikvb < ts_b; ++ikvb) {
            // 加载 kv block 到共享内存
            {
                uint64_t const end = (ikvb + 1) * bs;
                for (uint64_t ikv = ikvb * bs + it, i = it; ikv < end; ikv += bn, i += bn) {
                    kv_cache<T> cache = locate_cache<T>(pages_batch, kv_sbuf_b, kv_skv_b, kv_sh_b, bs, head / g, ikv);
                    for (uint64_t j = 0; j < d; ++j) {
                        kj[i * d + j] = cache.k[j];
                        vj[i * d + j] = cache.v[j];
                    }
                }
                __syncthreads();
            }
            // 加载 q_i 和 mask
            T *qi_val = new T[d];
            bool const *mask = mask_batch + iq * ts[b] * bs + ikvb * bs;

            for (uint64_t j = 0; j < d; ++j) {
                qi_val[j] = q[iq * d + j];
            }

            // 计算注意力分数
            T *scores = new T[d];
            T mi_local = -CUDART_INF_F;

            for (uint64_t i = 0; i < bs; ++i) {
                if (!mask[i]) {
                    scores[i] = -CUDART_INF_F;
                } else {
                    scores[i] = 0;
                    for (uint64_t j = 0; j < d; ++j) {
                        scores[i] += qi_val[j] * kj[i * d + j];
                    }
                    scores[i] *= scale;
                    if (scores[i] > mi_local) {
                        mi_local = scores[i];
                    }
                }
            }

            // 更新softmax状态
            T mi_new = max(mi, mi_local);
            T sum = 0.0;

            for (uint64_t i = 0; i < bs; ++i) {
                if (mask[i]) {
                    scores[i] = exp(scores[i] - mi_new);
                    sum += scores[i];
                } else {
                    scores[i] = 0;
                }
            }

            T exp_old = (mi == -CUDART_INF_F) ? 0.0 : exp(mi - mi_new);
            T li_new = exp_old * li + sum;
            T rdi = (li_new == 0) ? 0.0 : 1.0 / li_new;

            // 更新输出
            for (uint64_t j = 0; j < d; ++j) {
                T v_acc = 0.0;
                for (uint64_t i = 0; i < bs; ++i) {
                    if (mask[i]) {
                        v_acc += scores[i] * vj[i * d + j];
                    }
                }
                oi[j] = oi[j] * exp_old * li * rdi + v_acc * rdi;
            }
            mi = mi_new;
            li = li_new;
        }
    __syncthreads(); // 确保共享内存使用完成
        // 保存最终状态
        m_batch[head * n_b + iq] = mi;
        l_batch[head * n_b + iq] = li;
    }
}

extern "C" __global__ void __attn_f64(
    double *const **kv_pages,
    double *const *q_,
    double **o_,
    bool *const *mask_,
    double *const *m,
    double *const *l,
    uint64_t const *n,
    uint64_t const d,
    uint64_t const *ts,
    uint64_t const bs,
    uint64_t const g,
    int64_t const sq,
    int64_t const so,
    int64_t const *kv_sbuf,
    int64_t const *kv_skv,
    int64_t const *kv_sh,
    float const scale) {
    __attn<double>(
        kv_pages, q_, o_, mask_, m, l,
        n, d, ts, bs, g, sq, so,
        kv_sbuf, kv_skv, kv_sh, scale);
    // 仅由第一个线程执行打印
    // 只在第一个线程中打印（避免重复输出）
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // 打印标量参数
        printf("Scalar parameters:\n");
        printf("  d (uint64_t): value = %lu\n", d);
        printf("  bs (uint64_t): value = %lu\n", bs);
        printf("  g (uint64_t): value = %lu\n", g);
        printf("  sq (int64_t): value = %ld\n", sq);
        printf("  so (int64_t): value = %ld\n", so);
        printf("  scale (float): value = %f\n", scale);

        // 打印指针参数（地址和指向的第一个值）
        printf("\nPointer parameters:\n");

        // kv_pages (triple pointer)
        printf("kv_pages (addr: %p):\n", (void *)&kv_pages);
        if (kv_pages != nullptr) {
            printf("  *kv_pages (addr: %p)\n", (void *)*kv_pages);
            if (*kv_pages != nullptr) {
                printf("    **kv_pages (addr: %p)\n", (void *)(*kv_pages)[0]);
                if ((*kv_pages)[0] != nullptr) {
                    printf("      ***kv_pages[0][0] = %f\n", (*kv_pages)[0][0]);
                } else {
                    printf("      **kv_pages[0] is NULL\n");
                }
            } else {
                printf("    *kv_pages is NULL\n");
            }
        } else {
            printf("  kv_pages is NULL\n");
        }

        // q_ (double pointer)
        printf("q_ (addr: %p):\n", (void *)&q_);
        if (q_ != nullptr) {
            printf("  *q_ (addr: %p)\n", (void *)*q_);
            if (*q_ != nullptr) {
                printf("    **q_ = %f\n", (*q_)[0]);
            } else {
                printf("    *q_ is NULL\n");
            }
        } else {
            printf("  q_ is NULL\n");
        }

        // o_ (double pointer)
        printf("o_ (addr: %p):\n", (void *)&o_);
        if (o_ != nullptr) {
            printf("  *o_ (addr: %p)\n", (void *)*o_);
            if (*o_ != nullptr) {
                printf("    **o_ = %f\n", (*o_)[0]);
            } else {
                printf("    *o_ is NULL\n");
            }
        } else {
            printf("  o_ is NULL\n");
        }

        // mask_ (double pointer to bool)
        printf("mask_ (addr: %p):\n", (void *)&mask_);
        if (mask_ != nullptr) {
            printf("  *mask_ (addr: %p)\n", (void *)*mask_);
            if (*mask_ != nullptr) {
                printf("    **mask_ = %d\n", (int)(*mask_)[0]);
            } else {
                printf("    *mask_ is NULL\n");
            }
        } else {
            printf("  mask_ is NULL\n");
        }

        // m (double pointer)
        printf("m (addr: %p):\n", (void *)&m);
        if (m != nullptr) {
            printf("  *m (addr: %p)\n", (void *)*m);
            if (*m != nullptr) {
                printf("    **m = %f\n", (*m)[0]);
            } else {
                printf("    *m is NULL\n");
            }
        } else {
            printf("  m is NULL\n");
        }

        // l (double pointer)
        printf("l (addr: %p):\n", (void *)&l);
        if (l != nullptr) {
            printf("  *l (addr: %p)\n", (void *)*l);
            if (*l != nullptr) {
                printf("    **l = %f\n", (*l)[0]);
            } else {
                printf("    *l is NULL\n");
            }
        } else {
            printf("  l is NULL\n");
        }

        // n (pointer to uint64_t)
        printf("n (addr: %p):\n", (void *)&n);
        if (n != nullptr) {
            printf("  *n = %lu\n", *n);
        } else {
            printf("  n is NULL\n");
        }

        // ts (pointer to uint64_t)
        printf("ts (addr: %p):\n", (void *)&ts);
        if (ts != nullptr) {
            printf("  *ts = %lu\n", *ts);
        } else {
            printf("  ts is NULL\n");
        }

        // kv_sbuf (pointer to int64_t)
        printf("kv_sbuf (addr: %p):\n", (void *)&kv_sbuf);
        if (kv_sbuf != nullptr) {
            printf("  *kv_sbuf = %ld\n", *kv_sbuf);
        } else {
            printf("  kv_sbuf is NULL\n");
        }

        // kv_skv (pointer to int64_t)
        printf("kv_skv (addr: %p):\n", (void *)&kv_skv);
        if (kv_skv != nullptr) {
            printf("  *kv_skv = %ld\n", *kv_skv);
        } else {
            printf("  kv_skv is NULL\n");
        }

        // kv_sh (pointer to int64_t)
        printf("kv_sh (addr: %p):\n", (void *)&kv_sh);
        if (kv_sh != nullptr) {
            printf("  *kv_sh = %ld\n", *kv_sh);
        } else {
            printf("  kv_sh is NULL\n");
        }
    }
}
int main() {
    return 0;
}
