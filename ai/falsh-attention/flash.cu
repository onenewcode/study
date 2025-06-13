#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void forward_kernel(const half *Q, const half *K, const half *V, const int N, const int d,
                               const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                               float *l, float *m, half *O)
{
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y; // batch and head index

    // Offset into Q,K,V,O,l,m - different for each batch and head
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d); // gridDim.y = nh
    int lm_offset = (bx * gridDim.y * N) + (by * N);          // offset for l and m

    // Define SRAM for Q,K,V,S (use float for accumulation)
    extern __shared__ float sram[];
    int tile_size = Bc * d;
    float *Qi = sram;
    float *Kj = &sram[tile_size];
    float *Vj = &sram[tile_size * 2];
    float *S = &sram[tile_size * 3];

    // Temporary buffer for half->float conversion
    for (int j = 0; j < Tc; j++)
    {
        // Load Kj, Vj to SRAM (convert half to float)
        for (int x = 0; x < d; x++)
        {
            Kj[(tx * d) + x] = __half2float(K[qkv_offset + (tile_size * j) + (tx * d) + x]);
            Vj[(tx * d) + x] = __half2float(V[qkv_offset + (tile_size * j) + (tx * d) + x]);
        }
        __syncthreads();

        for (int i = 0; i < Tr; i++)
        {
            // Load Qi to SRAM, l and m to registers
            for (int x = 0; x < d; x++)
            {
                Qi[(tx * d) + x] = __half2float(Q[qkv_offset + (tile_size * i) + (tx * d) + x]);
            }
            float row_m_prev = m[lm_offset + (Br * i) + tx];
            float row_l_prev = l[lm_offset + (Br * i) + tx];

            // S = QK^T, row_m = rowmax(S)
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++)
            {
                float sum = 0;
                for (int x = 0; x < d; x++)
                {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                S[(Bc * tx) + y] = sum;
                if (sum > row_m)
                    row_m = sum;
            }

            // P = exp(S - row_m), row_l = rowsum(P)
            float row_l = 0;
            for (int y = 0; y < Bc; y++)
            {
                S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
                row_l += S[(Bc * tx) + y];
            }

            // Compute new m and l
            float row_m_new = max(row_m_prev, row_m);
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

            // Write O, l, m to HBM
            for (int x = 0; x < d; x++)
            {
                float pv = 0; // Pij * Vj
                for (int y = 0; y < Bc; y++)
                {
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                }
                float out_val = (1 / row_l_new) * ((row_l_prev * __expf(row_m_prev - row_m_new) * __half2float(O[qkv_offset + (tile_size * i) + (tx * d) + x])) + (__expf(row_m - row_m_new) * pv));
                O[qkv_offset + (tile_size * i) + (tx * d) + x] = __float2half(out_val);
            }
            m[lm_offset + (Br * i) + tx] = row_m_new;
            l[lm_offset + (Br * i) + tx] = row_l_new;
        }
        __syncthreads();
    }
}

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V)
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

    forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        reinterpret_cast<half *>(Q.data_ptr<at::Half>()),
        reinterpret_cast<half *>(K.data_ptr<at::Half>()),
        reinterpret_cast<half *>(V.data_ptr<at::Half>()),
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        l.data_ptr<float>(), m.data_ptr<float>(),
        reinterpret_cast<half *>(O.data_ptr<at::Half>())
    );
    return O;
}
