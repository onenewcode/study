#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
__global__ void forward_kernel2(const half *Q, const half *K, const half *V, const int N, const int d,
                               const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                               float *l, float *m, half *O)
{
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y; // batch and head index

    // Offset into Q,K,V,O,l,m - different for each batch and head
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d); // gridDim.y = nh
    int lm_offset = (bx * gridDim.y * N) + (by * N);          // offset for l and m

    // Define SRAM for Q,K,V,S (全部用half存储)
    extern __shared__ half sram[];
    int tile_size = Bc * d;
    half *Qi = sram;
    half *Kj = &sram[tile_size];
    half *Vj = &sram[tile_size * 2];
    half *S = &sram[tile_size * 3];

    // Temporary buffer for half->float conversion only for accumulation
    for (int j = 0; j < Tc; j++)
    {
        // Load Kj, Vj to SRAM (向量化float4加载，每次8个half)
        half2 *Kj2 = reinterpret_cast<half2*>(Kj);
        half2 *Vj2 = reinterpret_cast<half2*>(Vj);
        const half2 *K2 = reinterpret_cast<const half2*>(K);
        const half2 *V2 = reinterpret_cast<const half2*>(V);
        int d2 = d / 2;
        for (int x = 0; x < d2; x++)
        {
            Kj2[tx * d2 + x] = K2[(qkv_offset + (tile_size * j) + (tx * d) + 2 * x) / 2];
            Vj2[tx * d2 + x] = V2[(qkv_offset + (tile_size * j) + (tx * d) + 2 * x) / 2];
        }
        __syncthreads();

        for (int i = 0; i < Tr; i++)
        {
            // Load Qi to SRAM (向量化float4加载，每次8个half)
            half2 *Qi2 = reinterpret_cast<half2*>(Qi);
            const half2 *Q2 = reinterpret_cast<const half2*>(Q);
            int d2 = d / 2;
            for (int x = 0; x < d2; x++) {
                Qi2[tx * d2 + x] = Q2[(qkv_offset + (tile_size * i) + (tx * d) + 2 * x) / 2];
            }
            float row_m_prev = m[lm_offset + (Br * i) + tx];
            float row_l_prev = l[lm_offset + (Br * i) + tx];

            // S = QK^T, row_m = rowmax(S)
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++)
            {
                float sum = 0.0f;
                for (int x = 0; x < d; x++)
                {
                    sum += __half2float(Qi[(tx * d) + x]) * __half2float(Kj[(y * d) + x]);
                }
                sum *= softmax_scale;
                S[(Bc * tx) + y] = __float2half(sum);
                if (sum > row_m)
                    row_m = sum;
            }

            // P = exp(S - row_m), row_l = rowsum(P)
            half row_l = __float2half(0.0f);
            for (int y = 0; y < Bc; y++)
            {
                S[(Bc * tx) + y] = hexp(__hsub(S[(Bc * tx) + y], __float2half(row_m)));
                row_l = __hadd(row_l, S[(Bc * tx) + y]);
            }

            // Compute new m and l (float for stability)
            float row_m_new = max(row_m_prev, row_m);
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * __half2float(row_l));

            // Write O, l, m to HBM
            for (int x = 0; x < d; x++)
            {
                half pv = __float2half(0.0f); // Pij * Vj
                for (int y = 0; y < Bc; y++)
                {
                    pv = __hadd(pv, __hmul(S[(Bc * tx) + y], Vj[(y * d) + x]));
                }
                float out_val = (1.0f / row_l_new) * ((row_l_prev * __expf(row_m_prev - row_m_new) * __half2float(O[qkv_offset + (tile_size * i) + (tx * d) + x])) + (__expf(row_m - row_m_new) * __half2float(pv)));
                O[qkv_offset + (tile_size * i) + (tx * d) + x] = __float2half(out_val);
            }
            m[lm_offset + (Br * i) + tx] = row_m_new;
            l[lm_offset + (Br * i) + tx] = row_l_new;
        }
        __syncthreads();
    }
}

torch::Tensor forward2(torch::Tensor Q, torch::Tensor K, torch::Tensor V)
{
    // TODO: determine Bc, Br dynamically
    const int Bc = 32;
    const int Br = 32;

    // Ensure input is half
    Q = Q.to(torch::kHalf);
    K = K.to(torch::kHalf);
    V = V.to(torch::kHalf);

    const int B = Q.size(0);
    const int nh = Q.size(1);
    const int N = Q.size(2);
    const int d = Q.size(3);

    const int Tc = ceil((float)N / Bc);
    const int Tr = ceil((float)N / Br);
    const float softmax_scale = 1.0f / sqrtf((float)d);

    // Initialize O, l, m to HBM
    auto O = torch::zeros_like(Q);
    auto l = torch::zeros({B, nh, N}, torch::dtype(torch::kFloat).device(Q.device()));
    auto m = torch::full({B, nh, N}, -INFINITY, torch::dtype(torch::kFloat).device(Q.device()));

    // Calculate SRAM size needed per block
    const int sram_size = (3 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    dim3 grid_dim(B, nh); // batch_size x num_heads
    dim3 block_dim(Bc);   // Bc threads per block

    forward_kernel2<<<grid_dim, block_dim, sram_size>>>(
        reinterpret_cast<half *>(Q.data_ptr<at::Half>()),
        reinterpret_cast<half *>(K.data_ptr<at::Half>()),
        reinterpret_cast<half *>(V.data_ptr<at::Half>()),
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        l.data_ptr<float>(), m.data_ptr<float>(),
        reinterpret_cast<half *>(O.data_ptr<at::Half>())
    );
    return O;
}
