#include <cuda_runtime.h>
#include <iostream>
#include <cfloat>
#include <vector>
__global__ void conv2d_kernel(const float* input, const float* kernel, float* output,
    int in_width, int in_height,
    int kernel_size,
    int out_width, int out_height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= out_width || y >= out_height) return;

    float sum = 0.0f;

    for (int ky = 0; ky < kernel_size; ++ky) {
        for (int kx = 0; kx < kernel_size; ++kx) {

            int in_x = x + kx;
            int in_y = y + ky;

            int input_idx = in_y * in_width + in_x;
            int kernel_idx = ky * kernel_size + kx;

            sum += input[input_idx] * kernel[kernel_idx];
        }
    }

    output[y * out_width + x] = sum;
}

extern "C" void launchConv2D(const float* d_input, const float* d_weight, float* d_output,
    int in_w, int in_h, int k_size, int out_w, int out_h) {

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((out_w + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (out_h + threadsPerBlock.y - 1) / threadsPerBlock.y);
    conv2d_kernel << <numBlocks, threadsPerBlock >> > (d_input, d_weight, d_output,
        in_w, in_h, k_size, out_w, out_h);

    cudaDeviceSynchronize();
}

__global__ void addBiasRelu_kernel(float* data, const float* bias, int channel_size, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_size) return;

    int channel = idx / channel_size;

    float val = data[idx] + bias[channel];

    if (val < 0.0f) {
        val = 0.0f;
    }

    data[idx] = val;
}

extern "C" void launchAddBiasRelu(float* d_data, const float* d_bias, int channel_size, int num_channels) {
    int total_size = channel_size * num_channels;

    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;

    addBiasRelu_kernel << <blocks, threads >> > (d_data, d_bias, channel_size, total_size);
    cudaDeviceSynchronize();
}

__global__ void maxpool_kernel(const float* input, float* output,
    int in_w, int in_h,
    int out_w, int out_h) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y; 

 
    if (x >= out_w || y >= out_h) return;
    int start_x = x * 2;
    int start_y = y * 2;

    float max_val = -FLT_MAX; // とりあえずすごく小さい値

    for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {

            int cur_x = start_x + dx;
            int cur_y = start_y + dy;

            if (cur_x < in_w && cur_y < in_h) {
                float val = input[cur_y * in_w + cur_x];
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
    }

    output[y * out_w + x] = max_val;
}

extern "C" void launchMaxPool(const float* d_input, float* d_output,
    int in_w, int in_h, int out_w, int out_h) {

    dim3 threads(16, 16);
    dim3 blocks((out_w + threads.x - 1) / threads.x,
        (out_h + threads.y - 1) / threads.y);

    maxpool_kernel << <blocks, threads >> > (d_input, d_output, in_w, in_h, out_w, out_h);
    cudaDeviceSynchronize();
}

// ============================================================
// 第2層以降用: マルチチャンネル畳み込みカーネル
// 複数の入力チャンネルを同時に処理して1つの出力ピクセルを計算
// ============================================================
__global__ void conv2d_multichannel_kernel(
    const float* input,
    const float* weights, 
    float* output,          
    int in_w, int in_h,
    int in_channels,
    int kernel_size,
    int out_w, int out_h,
    int out_channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;  
    int y = blockIdx.y * blockDim.y + threadIdx.y; 
    int filter_idx = blockIdx.z;                     

    if (x >= out_w || y >= out_h || filter_idx >= out_channels) return;

    float sum = 0.0f;

    for (int c = 0; c < in_channels; ++c) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int in_x = x + kx;
                int in_y = y + ky;

                int input_idx = c * (in_w * in_h) + in_y * in_w + in_x;

                int weight_idx = filter_idx * (in_channels * kernel_size * kernel_size) +
                                 c * (kernel_size * kernel_size) +
                                 ky * kernel_size + kx;

                sum += input[input_idx] * weights[weight_idx];
            }
        }
    }

    output[filter_idx * (out_w * out_h) + y * out_w + x] = sum;
}

extern "C" void launchConv2DMultiChannel(
    const float* d_input, const float* d_weights, float* d_output,
    int in_w, int in_h, int in_channels,
    int kernel_size, int out_w, int out_h, int out_channels)
{
    dim3 threads(16, 16, 1);
    dim3 blocks(
        (out_w + threads.x - 1) / threads.x,
        (out_h + threads.y - 1) / threads.y,
        out_channels 
    );

    conv2d_multichannel_kernel<<<blocks, threads>>>(
        d_input, d_weights, d_output,
        in_w, in_h, in_channels,
        kernel_size, out_w, out_h, out_channels);

    cudaDeviceSynchronize();
}


extern "C" void launchMaxPoolMultiChannel(
    const float* d_input, float* d_output,
    int in_w, int in_h, int out_w, int out_h, int num_channels)
{
    int in_channel_size = in_w * in_h;
    int out_channel_size = out_w * out_h;

    for (int c = 0; c < num_channels; ++c) {
        const float* channel_input = d_input + c * in_channel_size;
        float* channel_output = d_output + c * out_channel_size;

        dim3 threads(16, 16);
        dim3 blocks((out_w + threads.x - 1) / threads.x,
                    (out_h + threads.y - 1) / threads.y);

        maxpool_kernel<<<blocks, threads>>>(channel_input, channel_output, in_w, in_h, out_w, out_h);
    }
    cudaDeviceSynchronize();
}


__global__ void fc_kernel(
    const float* input,     
    const float* weights,   
    const float* bias,      
    float* output,         
    int in_features,
    int out_features)
{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (out_idx >= out_features) return;
    
    float sum = 0.0f;
    
    for (int i = 0; i < in_features; ++i) {

        sum += input[i] * weights[out_idx * in_features + i];
    }
    

    sum += bias[out_idx];
    
    output[out_idx] = sum;
}


extern "C" void launchFullyConnected(
    const float* d_input, const float* d_weights, const float* d_bias, float* d_output,
    int in_features, int out_features)
{
    int threads = 256;
    int blocks = (out_features + threads - 1) / threads;
    
    fc_kernel<<<blocks, threads>>>(d_input, d_weights, d_bias, d_output, in_features, out_features);
    cudaDeviceSynchronize();
}


// ============================================================
// Softmax実装: GPU上で完全に実行
// ============================================================

// リダクション用: 配列の最大値を求める(単一ブロック用)
__global__ void find_max_kernel(const float* input, float* max_val, int size) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = tid;
    
    // データをシェアードメモリにロード
    sdata[tid] = (i < size) ? input[i] : -FLT_MAX;
    __syncthreads();
    
    // リダクション
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    // 結果を書き込み
    if (tid == 0) {
        *max_val = sdata[0];
    }
}

// リダクション用: exp(x - max)の合計を求める(単一ブロック用)
__global__ void sum_exp_kernel(const float* input, const float* max_val, float* sum_exp, int size) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = tid;
    
    // exp(x - max)を計算してシェアードメモリにロード
    sdata[tid] = (i < size) ? expf(input[i] - *max_val) : 0.0f;
    __syncthreads();
    
    // リダクション
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // 結果を書き込み
    if (tid == 0) {
        *sum_exp = sdata[0];
    }
}

// 最終的な正規化
__global__ void softmax_normalize_kernel(const float* input, float* output, const float* max_val, const float* sum_exp, int size) {
    int idx = threadIdx.x;
    if (idx >= size) return;
    
    output[idx] = expf(input[idx] - *max_val) / *sum_exp;
}

extern "C" void launchSoftmax(const float* d_input, float* d_output, int size) {
    // 小規模データ(size=10)なので単一ブロックで処理
    const int threads = (size + 31) & ~31; // 32の倍数に切り上げ(最低32)
    
    // 一時バッファ(GPUメモリ)
    float *d_max_val, *d_sum_exp;
    cudaMalloc(&d_max_val, sizeof(float));
    cudaMalloc(&d_sum_exp, sizeof(float));
    
    // ステップ1: 最大値を求める
    find_max_kernel<<<1, threads, threads * sizeof(float)>>>(d_input, d_max_val, size);
    
    // ステップ2: exp(x - max)の合計を求める
    sum_exp_kernel<<<1, threads, threads * sizeof(float)>>>(d_input, d_max_val, d_sum_exp, size);
    
    // ステップ3: 最終的な正規化
    softmax_normalize_kernel<<<1, threads>>>(d_input, d_output, d_max_val, d_sum_exp, size);
    
    cudaDeviceSynchronize();
    
    // クリーンアップ
    cudaFree(d_max_val);
    cudaFree(d_sum_exp);
}


