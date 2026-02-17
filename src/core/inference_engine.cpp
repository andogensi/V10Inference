#include "../../include/inference_engine.h"
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

// CUDAカーネル
extern "C" void launchConv2D(const float* d_input, const float* d_weight, float* d_output, 
                             int in_w, int in_h, int k_size, int out_w, int out_h);
extern "C" void launchAddBiasRelu(float* d_data, const float* d_bias, int channel_size, int num_channels);
extern "C" void launchMaxPool(const float* d_input, float* d_output, int in_w, int in_h, int out_w, int out_h);
extern "C" void launchConv2DMultiChannel(const float* d_input, const float* d_weights, float* d_output, int in_w, int in_h, int in_channels, int kernel_size, int out_w, int out_h, int out_channels);
extern "C" void launchMaxPoolMultiChannel(const float* d_input, float* d_output, int in_w, int in_h, 
                                          int out_w, int out_h, int num_channels);
extern "C" void launchFullyConnected(const float* d_input, const float* d_weights, const float* d_bias, 
                                     float* d_output, int in_features, int out_features);
extern "C" void launchSoftmax(const float* d_input, float* d_output, int size);

InferenceEngine::InferenceEngine(const ModelLoader& model) : model_(model) {
}

InferenceEngine::~InferenceEngine() {
    freeGPU();
}

bool InferenceEngine::allocateGPU() {
    if (gpu_allocated_) return true;

    std::vector<float> h_l1_weights = model_.getTensorData("Parameter5");
    std::vector<float> h_l1_bias    = model_.getTensorData("Parameter6");
    std::vector<float> h_l2_weights = model_.getTensorData("Parameter87");
    std::vector<float> h_l2_bias    = model_.getTensorData("Parameter88");
    std::vector<float> h_l3_weights = model_.getTensorData("Parameter193");
    std::vector<float> h_l3_bias    = model_.getTensorData("Parameter194");

    if (h_l1_weights.empty() || h_l1_bias.empty() ||
        h_l2_weights.empty() || h_l2_bias.empty() ||
        h_l3_weights.empty() || h_l3_bias.empty()) {
        std::cerr << "Error: Model parameters not found!" << std::endl;
        return false;
    }
    cudaMalloc(&d_input_, 28 * 28 * sizeof(float));

   // Layer 1
    cudaMalloc(&d_l1_weights_, h_l1_weights.size() * sizeof(float));
    cudaMalloc(&d_l1_bias_, h_l1_bias.size() * sizeof(float));
    cudaMalloc(&d_l1_conv_out_, 8 * 24 * 24 * sizeof(float));
    cudaMalloc(&d_l1_pool_out_, 8 * 12 * 12 * sizeof(float));

    // Layer 2;
    cudaMalloc(&d_l2_weights_, h_l2_weights.size() * sizeof(float));
    cudaMalloc(&d_l2_bias_, h_l2_bias.size() * sizeof(float));
    cudaMalloc(&d_l2_conv_out_, 16 * 8 * 8 * sizeof(float));
    cudaMalloc(&d_l2_pool_out_, 16 * 4 * 4 * sizeof(float));

    // Layer 3
    cudaMalloc(&d_l3_weights_, h_l3_weights.size() * sizeof(float));
    cudaMalloc(&d_l3_bias_, h_l3_bias.size() * sizeof(float));
    cudaMalloc(&d_l3_fc_out_, 10 * sizeof(float));
    cudaMalloc(&d_l3_smax_out_, 10 * sizeof(float));
    
    //一括で転送で60msくらい削減
    cudaMemcpy(d_l1_weights_, h_l1_weights.data(), h_l1_weights.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_l1_bias_, h_l1_bias.data(), h_l1_bias.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_l2_weights_, h_l2_weights.data(), h_l2_weights.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_l2_bias_, h_l2_bias.data(), h_l2_bias.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_l3_weights_, h_l3_weights.data(), h_l3_weights.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_l3_bias_, h_l3_bias.data(), h_l3_bias.size() * sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    gpu_allocated_ = true;
    return true;
}

void InferenceEngine::freeGPU() {
    if (!gpu_allocated_) return;

    cudaFree(d_input_);
    cudaFree(d_l1_weights_); cudaFree(d_l1_bias_);
    cudaFree(d_l1_conv_out_); cudaFree(d_l1_pool_out_);
    cudaFree(d_l2_weights_); cudaFree(d_l2_bias_);
    cudaFree(d_l2_conv_out_); cudaFree(d_l2_pool_out_);
    cudaFree(d_l3_weights_); cudaFree(d_l3_bias_);
    cudaFree(d_l3_fc_out_); cudaFree(d_l3_smax_out_);

    d_input_ = nullptr;
    d_l1_weights_ = nullptr; d_l1_bias_ = nullptr;
    d_l1_conv_out_ = nullptr; d_l1_pool_out_ = nullptr;
    d_l2_weights_ = nullptr; d_l2_bias_ = nullptr;
    d_l2_conv_out_ = nullptr; d_l2_pool_out_ = nullptr;
    d_l3_weights_ = nullptr; d_l3_bias_ = nullptr;
    d_l3_fc_out_ = nullptr; d_l3_smax_out_ = nullptr;

    gpu_allocated_ = false;
}

int InferenceEngine::run(const std::vector<float>& input_image) {
    if (!gpu_allocated_) {
        if (!allocateGPU()) return -1;
    }

    // 第1層
    auto t1_start = std::chrono::high_resolution_clock::now();
    runLayer1(input_image);
    cudaDeviceSynchronize();
    auto t1_end = std::chrono::high_resolution_clock::now();
    layer1_time_ms = std::chrono::duration<double, std::milli>(t1_end - t1_start).count();

    // 第2層
    auto t2_start = std::chrono::high_resolution_clock::now();
    runLayer2();
    cudaDeviceSynchronize();
    auto t2_end = std::chrono::high_resolution_clock::now();
    layer2_time_ms = std::chrono::duration<double, std::milli>(t2_end - t2_start).count();

    // 第3層
    auto t3_start = std::chrono::high_resolution_clock::now();
    int prediction = runLayer3();
    cudaDeviceSynchronize();
    auto t3_end = std::chrono::high_resolution_clock::now();
    layer3_time_ms = std::chrono::duration<double, std::milli>(t3_end - t3_start).count();

    return prediction;
}

void InferenceEngine::runLayer1(const std::vector<float>& input) {

    const int input_w = 28;
    const int input_h = 28;
    const int kernel_size = 5;
    const int num_filters = 8;
    const int output_w = input_w - kernel_size + 1; // 24
    const int output_h = input_h - kernel_size + 1; // 24
    const int ch_pix = output_w * output_h; // 576

    auto mem_start = std::chrono::high_resolution_clock::now();
    //画像だけ
    cudaMemcpy(d_input_, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
    auto mem_end = std::chrono::high_resolution_clock::now();
    double mem_time = std::chrono::duration<double, std::milli>(mem_end - mem_start).count();

    auto compute_start = std::chrono::high_resolution_clock::now();
    //8回の呼び出しを１度に
    launchConv2DMultiChannel(d_input_, d_l1_weights_, d_l1_conv_out_,
                             input_w, input_h, 1, kernel_size, output_w, output_h, num_filters);

    // Bias + ReLU
    launchAddBiasRelu(d_l1_conv_out_, d_l1_bias_, ch_pix, num_filters);
    auto compute_end = std::chrono::high_resolution_clock::now();
    double compute_time = std::chrono::duration<double, std::milli>(compute_end - compute_start).count();

    std::cout << "  [L1 Detail] Memory transfer: " << mem_time << " ms, Compute: " << compute_time << " ms" << std::endl;

#ifdef DEBUG_PRINT
    std::vector<float> h_output(num_filters * ch_pix);
    cudaMemcpy(h_output.data(), d_l1_conv_out_, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
    printFilterResults(h_output, num_filters, ch_pix, "Conv + ReLU (24x24)");
#endif

    const int pw = output_w / 2; // 12
    const int ph = output_h / 2; // 12
    launchMaxPoolMultiChannel(d_l1_conv_out_, d_l1_pool_out_, output_w, output_h, pw, ph, num_filters);

#ifdef DEBUG_PRINT
    const int p_pix = pw * ph;
    std::vector<float> h_pout(num_filters * p_pix);
    cudaMemcpy(h_pout.data(), d_l1_pool_out_, h_pout.size() * sizeof(float), cudaMemcpyDeviceToHost);
    printFilterResults(h_pout, num_filters, p_pix, "MaxPool (12x12)");
#endif
}

void InferenceEngine::runLayer2() {

    const int in_w = 12;
    const int in_h = 12;
    const int in_channels = 8;
    const int kernel_size = 5;
    const int out_channels = 16;
    const int out_w = in_w - kernel_size + 1; // 8
    const int out_h = in_h - kernel_size + 1; // 8
    const int out_pix = out_w * out_h; // 64

    launchConv2DMultiChannel(d_l1_pool_out_, d_l2_weights_, d_l2_conv_out_,
                             in_w, in_h, in_channels, kernel_size, out_w, out_h, out_channels);

    launchAddBiasRelu(d_l2_conv_out_, d_l2_bias_, out_pix, out_channels);

#ifdef DEBUG_PRINT
    std::vector<float> h_output(out_channels * out_pix);
    cudaMemcpy(h_output.data(), d_l2_conv_out_, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
    printFilterResults(h_output, out_channels, out_pix, "Conv + ReLU (8x8)");
#endif

    const int pw = out_w / 2; // 4
    const int ph = out_h / 2; // 4

    launchMaxPoolMultiChannel(d_l2_conv_out_, d_l2_pool_out_, out_w, out_h, 
                              pw, ph, out_channels);
    
#ifdef DEBUG_PRINT
    const int p_pix = pw * ph;
    std::vector<float> h_pout(out_channels * p_pix);
    cudaMemcpy(h_pout.data(), d_l2_pool_out_, h_pout.size() * sizeof(float), cudaMemcpyDeviceToHost);
    printFilterResults(h_pout, out_channels, p_pix, "MaxPool (4x4)");
#endif
}

int InferenceEngine::runLayer3() {

    const int in_features = 16 * 16; 
    const int out_features = 10;

    launchFullyConnected(d_l2_pool_out_, d_l3_weights_, d_l3_bias_, d_l3_fc_out_, in_features, out_features);

#ifdef DEBUG_PRINT
    std::vector<float> h_output(out_features);
    cudaMemcpy(h_output.data(), d_l3_fc_out_, out_features * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "- F Output -" << std::endl;
    for (int i = 0; i < out_features; ++i) {
        std::cout << "Class [" << i << "]: " << h_output[i] << std::endl;
    }
#endif

    launchSoftmax(d_l3_fc_out_, d_l3_smax_out_, out_features);

    std::vector<float> h_smax(out_features);
    cudaMemcpy(h_smax.data(), d_l3_smax_out_, out_features * sizeof(float), cudaMemcpyDeviceToHost);

    float max_prob = 0.0f;
    int pred_cls = 0;
    
    for (int i = 0; i < out_features; ++i) {
        if (h_smax[i] > max_prob) {
            max_prob = h_smax[i];
            pred_cls = i;
        }
    }

    std::cout << "Predicton: " << pred_cls << " (Confidence: " << (max_prob * 100.0f) << "%)" << std::endl;

    return pred_cls;
}

void InferenceEngine::printFilterResults(const std::vector<float>& data, int num_filters, 
                                        int pix_per_filt, const std::string& label) {
    std::cout << "--- " << label << " ---" << std::endl;
    for (int i = 0; i < num_filters; ++i) {
        float val = data[i * pix_per_filt];
        std::cout << "Filter [" << i << "] Result[0][0]: " << val << std::endl;
    }
}
