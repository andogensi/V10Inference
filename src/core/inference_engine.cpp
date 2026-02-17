#include "../../include/inference_engine.h"
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

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
    createTimingEvents();
}

InferenceEngine::~InferenceEngine() {
    freeGPU();
    destroyTimingEvents();
}

bool InferenceEngine::allocateGPU() {
    if (gpu_allocated_) return true;

    cudaStreamCreate(&stream_);

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
    
    cudaMemcpyAsync(d_l1_weights_, h_l1_weights.data(), h_l1_weights.size() * sizeof(float), cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_l1_bias_, h_l1_bias.data(), h_l1_bias.size() * sizeof(float), cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_l2_weights_, h_l2_weights.data(), h_l2_weights.size() * sizeof(float), cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_l2_bias_, h_l2_bias.data(), h_l2_bias.size() * sizeof(float), cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_l3_weights_, h_l3_weights.data(), h_l3_weights.size() * sizeof(float), cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_l3_bias_, h_l3_bias.data(), h_l3_bias.size() * sizeof(float), cudaMemcpyHostToDevice, stream_);

    cudaStreamSynchronize(stream_);
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
    
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }

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
    cudaEventRecord(event_start_, stream_);
    runLayer1(input_image);
    cudaEventRecord(event_end_, stream_);
    cudaEventSynchronize(event_end_);
    float elapsed;
    cudaEventElapsedTime(&elapsed, event_start_, event_end_);
    layer1_time_ms = elapsed;

    // 第2層
    cudaEventRecord(event_start_, stream_);
    runLayer2();
    cudaEventRecord(event_end_, stream_);
    cudaEventSynchronize(event_end_);
    cudaEventElapsedTime(&elapsed, event_start_, event_end_);
    layer2_time_ms = elapsed;

    // 第3層
    cudaEventRecord(event_start_, stream_);
    int prediction = runLayer3();
    cudaEventRecord(event_end_, stream_);
    cudaEventSynchronize(event_end_);
    cudaEventElapsedTime(&elapsed, event_start_, event_end_);
    layer3_time_ms = elapsed;

    return prediction;
}

// ウォームアップ用）
int InferenceEngine::runSilent(const std::vector<float>& input_image) {
    if (!gpu_allocated_) {
        if (!allocateGPU()) return -1;
    }

    runLayer1Silent(input_image);
    runLayer2Silent();
    return runLayer3Silent();
}


void InferenceEngine::runLayer1Silent(const std::vector<float>& input) {
    const int input_w = 28, input_h = 28, kernel_size = 5, num_filters = 8;
    const int output_w = input_w - kernel_size + 1; // 24
    const int output_h = input_h - kernel_size + 1; // 24
    const int ch_pix = output_w * output_h; // 576

    cudaMemcpyAsync(d_input_, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice, stream_);
    launchConv2DMultiChannel(d_input_, d_l1_weights_, d_l1_conv_out_,
                             input_w, input_h, 1, kernel_size, output_w, output_h, num_filters);
    launchAddBiasRelu(d_l1_conv_out_, d_l1_bias_, ch_pix, num_filters);
    
    const int pw = output_w / 2; // 12
    const int ph = output_h / 2; // 12
    launchMaxPoolMultiChannel(d_l1_conv_out_, d_l1_pool_out_, output_w, output_h, pw, ph, num_filters);
}

void InferenceEngine::runLayer2Silent() {
    const int in_w = 12, in_h = 12, in_channels = 8, kernel_size = 5, out_channels = 16;
    const int out_w = in_w - kernel_size + 1; // 8
    const int out_h = in_h - kernel_size + 1; // 8
    const int out_pix = out_w * out_h; // 64

    launchConv2DMultiChannel(d_l1_pool_out_, d_l2_weights_, d_l2_conv_out_,
                             in_w, in_h, in_channels, kernel_size, out_w, out_h, out_channels);
    launchAddBiasRelu(d_l2_conv_out_, d_l2_bias_, out_pix, out_channels);
    
    const int pw = out_w / 2; // 4
    const int ph = out_h / 2; // 4
    launchMaxPoolMultiChannel(d_l2_conv_out_, d_l2_pool_out_, out_w, out_h, pw, ph, out_channels);
}

int InferenceEngine::runLayer3Silent() {
    const int in_features = 16 * 16; 
    const int out_features = 10;

    launchFullyConnected(d_l2_pool_out_, d_l3_weights_, d_l3_bias_, d_l3_fc_out_, in_features, out_features);
    launchSoftmax(d_l3_fc_out_, d_l3_smax_out_, out_features);

    // Transfer
    std::vector<float> h_smax(out_features);
    cudaMemcpyAsync(h_smax.data(), d_l3_smax_out_, out_features * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    float max_prob = 0.0f;
    int pred_cls = 0;
    for (int i = 0; i < out_features; ++i) {
        if (h_smax[i] > max_prob) {
            max_prob = h_smax[i];
            pred_cls = i;
        }
    }
    return pred_cls;
}

void InferenceEngine::runLayer1(const std::vector<float>& input) {

    const int input_w = 28;
    const int input_h = 28;
    const int kernel_size = 5;
    const int num_filters = 8;
    const int output_w = input_w - kernel_size + 1; // 24
    const int output_h = input_h - kernel_size + 1; // 24
    const int ch_pix = output_w * output_h; // 576

    float h2d_time = 0.0f, kernel_time = 0.0f;
    
    cudaEventRecord(event_h2d_start_, stream_);
    cudaMemcpyAsync(d_input_, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice, stream_);
    cudaEventRecord(event_h2d_end_, stream_);
    cudaEventSynchronize(event_h2d_end_);
    cudaEventElapsedTime(&h2d_time, event_h2d_start_, event_h2d_end_);
    last_l1_h2d_ = h2d_time;

    cudaEventRecord(event_kernel_start_, stream_);
    launchConv2DMultiChannel(d_input_, d_l1_weights_, d_l1_conv_out_,
                             input_w, input_h, 1, kernel_size, output_w, output_h, num_filters);
    launchAddBiasRelu(d_l1_conv_out_, d_l1_bias_, ch_pix, num_filters);
    cudaEventRecord(event_kernel_end_, stream_);
    cudaEventSynchronize(event_kernel_end_);
    cudaEventElapsedTime(&kernel_time, event_kernel_start_, event_kernel_end_);

    std::cout << "  [L1] H2D: " << h2d_time << " ms, KernelConv+ReLU: " << kernel_time << " ms";

#ifdef DEBUG_PRINT
    std::vector<float> h_output(num_filters * ch_pix);
    cudaMemcpy(h_output.data(), d_l1_conv_out_, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
    printFilterResults(h_output, num_filters, ch_pix, "Conv + ReLU 24");
#endif

 
    cudaEventRecord(event_kernel_start_, stream_);
    const int pw = output_w / 2; // 12
    const int ph = output_h / 2; // 12
    launchMaxPoolMultiChannel(d_l1_conv_out_, d_l1_pool_out_, output_w, output_h, pw, ph, num_filters);
    cudaEventRecord(event_kernel_end_, stream_);
    cudaEventSynchronize(event_kernel_end_);
    float pool_time = 0.0f;
    cudaEventElapsedTime(&pool_time, event_kernel_start_, event_kernel_end_);
    last_l1_kernel_ = kernel_time + pool_time;
    last_l1_d2h_ = 0.0;  // Layer1にはD2H転送はしない
    
    std::cout << ", Pool: " << pool_time << " ms" << std::endl;

#ifdef DEBUG_PRINT
    const int p_pix = pw * ph;
    std::vector<float> h_pout(num_filters * p_pix);
    cudaMemcpy(h_pout.data(), d_l1_pool_out_, h_pout.size() * sizeof(float), cudaMemcpyDeviceToHost);
    printFilterResults(h_pout, num_filters, p_pix, "MaxPool 12");
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

    float kernel_time = 0.0f;
    
    last_l2_h2d_ = 0.0;  
    
    cudaEventRecord(event_kernel_start_, stream_);
    launchConv2DMultiChannel(d_l1_pool_out_, d_l2_weights_, d_l2_conv_out_,
                             in_w, in_h, in_channels, kernel_size, out_w, out_h, out_channels);
    launchAddBiasRelu(d_l2_conv_out_, d_l2_bias_, out_pix, out_channels);
    cudaEventRecord(event_kernel_end_, stream_);
    cudaEventSynchronize(event_kernel_end_);
    cudaEventElapsedTime(&kernel_time, event_kernel_start_, event_kernel_end_);
    
    std::cout << "  [L2] Kernel(Conv+ReLU): " << kernel_time << " ms";

#ifdef DEBUG_PRINT
    std::vector<float> h_output(out_channels * out_pix);
    cudaMemcpy(h_output.data(), d_l2_conv_out_, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
    printFilterResults(h_output, out_channels, out_pix, "Conv + ReLU (8x8)");
#endif

    // MaxPool
    cudaEventRecord(event_kernel_start_, stream_);
    const int pw = out_w / 2; // 4
    const int ph = out_h / 2; // 4
    launchMaxPoolMultiChannel(d_l2_conv_out_, d_l2_pool_out_, out_w, out_h, 
                              pw, ph, out_channels);
    cudaEventRecord(event_kernel_end_, stream_);
    cudaEventSynchronize(event_kernel_end_);
    float pool_time = 0.0f;
    cudaEventElapsedTime(&pool_time, event_kernel_start_, event_kernel_end_);
    last_l2_kernel_ = kernel_time + pool_time;
    last_l2_d2h_ = 0.0; 
    
    std::cout << ", Pool: " << pool_time << " ms" << std::endl;
    
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

    float kernel_time = 0.0f, d2h_time = 0.0f;
    
    last_l3_h2d_ = 0.0;  

    cudaEventRecord(event_kernel_start_, stream_);
    launchFullyConnected(d_l2_pool_out_, d_l3_weights_, d_l3_bias_, d_l3_fc_out_, in_features, out_features);
    cudaEventRecord(event_kernel_end_, stream_);
    cudaEventSynchronize(event_kernel_end_);
    cudaEventElapsedTime(&kernel_time, event_kernel_start_, event_kernel_end_);

#ifdef DEBUG_PRINT
    std::vector<float> h_output(out_features);
    cudaMemcpy(h_output.data(), d_l3_fc_out_, out_features * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "- F Output -" << std::endl;
    for (int i = 0; i < out_features; ++i) {
        std::cout << "Class [" << i << "]: " << h_output[i] << std::endl;
    }
#endif

    // Softmax
    cudaEventRecord(event_kernel_start_, stream_);
    launchSoftmax(d_l3_fc_out_, d_l3_smax_out_, out_features);
    cudaEventRecord(event_kernel_end_, stream_);
    cudaEventSynchronize(event_kernel_end_);
    float softmax_time = 0.0f;
    cudaEventElapsedTime(&softmax_time, event_kernel_start_, event_kernel_end_);
    last_l3_kernel_ = kernel_time + softmax_time;

    // D2H Transfer (Async)
    std::vector<float> h_smax(out_features);
    cudaEventRecord(event_d2h_start_, stream_);
    cudaMemcpyAsync(h_smax.data(), d_l3_smax_out_, out_features * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    cudaEventRecord(event_d2h_end_, stream_);
    cudaEventSynchronize(event_d2h_end_);
    cudaEventElapsedTime(&d2h_time, event_d2h_start_, event_d2h_end_);
    last_l3_d2h_ = d2h_time;
    
    std::cout << "  [L3] Kernel(FC): " << kernel_time << " ms, Softmax: " << softmax_time 
              << " ms, D2H: " << d2h_time << " ms" << std::endl;

    float max_prob = 0.0f;
    int pred_cls = 0;
    
    for (int i = 0; i < out_features; ++i) {
        if (h_smax[i] > max_prob) {
            max_prob = h_smax[i];
            pred_cls = i;
        }
    }

    std::cout << "Prediction: " << pred_cls << " (Confidence: " << (max_prob * 100.0f) << "%)" << std::endl;

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

// CUDA Even
void InferenceEngine::createTimingEvents() {
    if (events_created_) return;
    
    cudaEventCreate(&event_start_);
    cudaEventCreate(&event_end_);
    cudaEventCreate(&event_h2d_start_);
    cudaEventCreate(&event_h2d_end_);
    cudaEventCreate(&event_kernel_start_);
    cudaEventCreate(&event_kernel_end_);
    cudaEventCreate(&event_d2h_start_);
    cudaEventCreate(&event_d2h_end_);
    cudaEventCreate(&event_e2e_start_);
    cudaEventCreate(&event_e2e_end_);
    cudaEventCreate(&event_l1_start_);
    cudaEventCreate(&event_l1_end_);
    cudaEventCreate(&event_l2_start_);
    cudaEventCreate(&event_l2_end_);
    cudaEventCreate(&event_l3_start_);
    cudaEventCreate(&event_l3_end_);
    
    events_created_ = true;
}

// 破棄
void InferenceEngine::destroyTimingEvents() {
    if (!events_created_) return;
    
    cudaEventDestroy(event_start_);
    cudaEventDestroy(event_end_);
    cudaEventDestroy(event_h2d_start_);
    cudaEventDestroy(event_h2d_end_);
    cudaEventDestroy(event_kernel_start_);
    cudaEventDestroy(event_kernel_end_);
    cudaEventDestroy(event_d2h_start_);
    cudaEventDestroy(event_d2h_end_);
    cudaEventDestroy(event_e2e_start_);
    cudaEventDestroy(event_e2e_end_);
    // カーネル計測用
    cudaEventDestroy(event_l1_start_);
    cudaEventDestroy(event_l1_end_);
    cudaEventDestroy(event_l2_start_);
    cudaEventDestroy(event_l2_end_);
    cudaEventDestroy(event_l3_start_);
    cudaEventDestroy(event_l3_end_);
    
    events_created_ = false;
}

double InferenceEngine::computePercentile(std::vector<double> sorted_times, double percentile) {
    if (sorted_times.empty()) return 0.0;
    std::sort(sorted_times.begin(), sorted_times.end());
    size_t idx = static_cast<size_t>(percentile / 100.0 * (sorted_times.size() - 1));
    return sorted_times[idx];
}
TimingStats InferenceEngine::computeStats(const std::vector<double>& times) {
    TimingStats stats;
    stats.num_samples = static_cast<int>(times.size());
    
    if (times.empty()) return stats;
    
    // 平均
    double sum = 0.0;
    for (double t : times) {
        sum += t;
    }
    stats.mean_ms = sum / times.size();
    
    //偏差
    double variance = 0.0;
    for (double t : times) {
        double diff = t - stats.mean_ms;
        variance += diff * diff;
    }
    stats.std_dev_ms = std::sqrt(variance / times.size());
    
    // 最小・最大
    stats.min_ms = *std::min_element(times.begin(), times.end());
    stats.max_ms = *std::max_element(times.begin(), times.end());
    
    // p50（中央値）とp95
    std::vector<double> sorted_times = times;
    std::sort(sorted_times.begin(), sorted_times.end());
    stats.p50_ms = computePercentile(sorted_times, 50.0);
    stats.p95_ms = computePercentile(sorted_times, 95.0);
    
    return stats;
}

// タイミング統計を表示
void InferenceEngine::printTimingStats() const {
    // GPU
    auto printStatsMs = [](const std::string& name, const TimingStats& stats) {
        std::cout << "  " << name << ": " 
                  << stats.mean_ms << " +/- " << stats.std_dev_ms << " ms"
                  << " [min=" << stats.min_ms << ", max=" << stats.max_ms 
                  << ", p50=" << stats.p50_ms << ", p95=" << stats.p95_ms << "]"
                  << std::endl;
    };
    
    // CPU
    auto printStatsUs = [](const std::string& name, const TimingStats& stats) {
        std::cout << "  " << name << ": " 
                  << (stats.mean_ms * 1000.0) << " +/- " << (stats.std_dev_ms * 1000.0) << " us"
                  << " [min=" << (stats.min_ms * 1000.0) << ", max=" << (stats.max_ms * 1000.0) 
                  << ", p50=" << (stats.p50_ms * 1000.0) << ", p95=" << (stats.p95_ms * 1000.0) << "]"
                  << std::endl;
    };
    
    std::cout << "\n[Phase Breakdown]" << std::endl;
    printStatsUs("Host Preprocessing ", phase_stats_.host_preprocess);
    printStatsMs("GPU Transfer (H2D) ", phase_stats_.gpu_transfer_h2d);
    printStatsMs("GPU Kernel         ", phase_stats_.gpu_kernel);
    printStatsMs("GPU Transfer (D2H) ", phase_stats_.gpu_transfer_d2h);
    printStatsUs("Host Postprocessing", phase_stats_.host_postprocess);
    printStatsUs("cudaStreamSync     ", phase_stats_.device_sync);
    
    std::cout << "\n[End-to-End]" << std::endl;
    printStatsMs("Total E2E          ", phase_stats_.e2e);
    
    std::cout << "\n[Layer Breakdown]" << std::endl;
    std::cout << "  Layer 1 (Conv+Pool):" << std::endl;
    std::cout << "    H2D:    " << layer1_stats_.h2d.mean_ms << " +/- " << layer1_stats_.h2d.std_dev_ms << " ms" << std::endl;
    std::cout << "    Kernel: " << layer1_stats_.kernel.mean_ms << " +/- " << layer1_stats_.kernel.std_dev_ms << " ms" << std::endl;
    std::cout << "    Total:  " << layer1_stats_.total.mean_ms << " +/- " << layer1_stats_.total.std_dev_ms << " ms" << std::endl;
    
    std::cout << "  Layer 2 (Conv+Pool):" << std::endl;
    std::cout << "    Kernel: " << layer2_stats_.kernel.mean_ms << " +/- " << layer2_stats_.kernel.std_dev_ms << " ms" << std::endl;
    std::cout << "    Total:  " << layer2_stats_.total.mean_ms << " +/- " << layer2_stats_.total.std_dev_ms << " ms" << std::endl;
    
    std::cout << "  Layer 3 (FC+Softmax):" << std::endl;
    std::cout << "    Kernel: " << layer3_stats_.kernel.mean_ms << " +/- " << layer3_stats_.kernel.std_dev_ms << " ms" << std::endl;
    std::cout << "    D2H:    " << layer3_stats_.d2h.mean_ms << " +/- " << layer3_stats_.d2h.std_dev_ms << " ms" << std::endl;
    std::cout << "    Total:  " << layer3_stats_.total.mean_ms << " +/- " << layer3_stats_.total.std_dev_ms << " ms" << std::endl;

}

// 複数実行
int InferenceEngine::runWithStats(const std::vector<float>& input_image, 
                                   int warmup_iterations, 
                                   int measure_iterations) {
    if (!gpu_allocated_) {
        if (!allocateGPU()) return -1;
    }
    
    const int input_w = 28, input_h = 28, kernel_size_l1 = 5, num_filters_l1 = 8;
    const int output_w_l1 = input_w - kernel_size_l1 + 1; // 24
    const int output_h_l1 = input_h - kernel_size_l1 + 1; // 24
    const int ch_pix_l1 = output_w_l1 * output_h_l1; // 576
    const int pw_l1 = output_w_l1 / 2; // 12
    const int ph_l1 = output_h_l1 / 2; // 12

    const int in_w_l2 = 12, in_h_l2 = 12, in_channels_l2 = 8, kernel_size_l2 = 5, out_channels_l2 = 16;
    const int out_w_l2 = in_w_l2 - kernel_size_l2 + 1; // 8
    const int out_h_l2 = in_h_l2 - kernel_size_l2 + 1; // 8
    const int out_pix_l2 = out_w_l2 * out_h_l2; // 64
    const int pw_l2 = out_w_l2 / 2; // 4
    const int ph_l2 = out_h_l2 / 2; // 4

    const int in_features_l3 = 16 * 16;
    const int out_features = 10;
    
    // ===== WARMUP PHASE =====
    std::cout << "Warmup: " << warmup_iterations << " iterations..." << std::flush;
    for (int i = 0; i < warmup_iterations; ++i) {
        runSilent(input_image);
    }
    cudaDeviceSynchronize();
    std::cout << " done." << std::endl;
    
    // ===== MEASUREMENT PHASE =====
    std::cout << "Measuring: " << measure_iterations << " iterations..." << std::flush;
    
    // 計測データ配列（ループ中はcoutしない）
    std::vector<double> e2e_times;
    std::vector<double> h2d_times, kernel_times, d2h_times;
    std::vector<double> host_pre_times, host_post_times, sync_times;
    std::vector<double> l1_kernel_times, l2_kernel_times, l3_kernel_times;
    
    e2e_times.reserve(measure_iterations);
    h2d_times.reserve(measure_iterations);
    kernel_times.reserve(measure_iterations);
    d2h_times.reserve(measure_iterations);
    host_pre_times.reserve(measure_iterations);
    host_post_times.reserve(measure_iterations);
    sync_times.reserve(measure_iterations);
    l1_kernel_times.reserve(measure_iterations);
    l2_kernel_times.reserve(measure_iterations);
    l3_kernel_times.reserve(measure_iterations);
    
    int final_prediction = -1;
    std::vector<float> h_smax(out_features);
    
    for (int iter = 0; iter < measure_iterations; ++iter) {
        auto host_start = std::chrono::high_resolution_clock::now();
        
        auto host_pre_end = std::chrono::high_resolution_clock::now();

        cudaEventRecord(event_e2e_start_, stream_);
 
        cudaEventRecord(event_h2d_start_, stream_);
        cudaMemcpyAsync(d_input_, input_image.data(), input_image.size() * sizeof(float), cudaMemcpyHostToDevice, stream_);
        cudaEventRecord(event_h2d_end_, stream_);
        

        cudaEventRecord(event_l1_start_, stream_);
        launchConv2DMultiChannel(d_input_, d_l1_weights_, d_l1_conv_out_,
                                 input_w, input_h, 1, kernel_size_l1, output_w_l1, output_h_l1, num_filters_l1);
        launchAddBiasRelu(d_l1_conv_out_, d_l1_bias_, ch_pix_l1, num_filters_l1);
        launchMaxPoolMultiChannel(d_l1_conv_out_, d_l1_pool_out_, output_w_l1, output_h_l1, pw_l1, ph_l1, num_filters_l1);
        cudaEventRecord(event_l1_end_, stream_);

        cudaEventRecord(event_l2_start_, stream_);
        launchConv2DMultiChannel(d_l1_pool_out_, d_l2_weights_, d_l2_conv_out_,
                                 in_w_l2, in_h_l2, in_channels_l2, kernel_size_l2, out_w_l2, out_h_l2, out_channels_l2);
        launchAddBiasRelu(d_l2_conv_out_, d_l2_bias_, out_pix_l2, out_channels_l2);
        launchMaxPoolMultiChannel(d_l2_conv_out_, d_l2_pool_out_, out_w_l2, out_h_l2, pw_l2, ph_l2, out_channels_l2);
        cudaEventRecord(event_l2_end_, stream_);
        cudaEventRecord(event_l3_start_, stream_);
        launchFullyConnected(d_l2_pool_out_, d_l3_weights_, d_l3_bias_, d_l3_fc_out_, in_features_l3, out_features);
        launchSoftmax(d_l3_fc_out_, d_l3_smax_out_, out_features);
        cudaEventRecord(event_l3_end_, stream_);
        
        // 全カーネル終了
        cudaEventRecord(event_kernel_end_, stream_);

        cudaEventRecord(event_d2h_start_, stream_);
        cudaMemcpyAsync(h_smax.data(), d_l3_smax_out_, out_features * sizeof(float), cudaMemcpyDeviceToHost, stream_);
        cudaEventRecord(event_d2h_end_, stream_);

        cudaEventRecord(event_e2e_end_, stream_);
        
        // 計測
        auto sync_start = std::chrono::high_resolution_clock::now();
        cudaStreamSynchronize(stream_);
        auto sync_end = std::chrono::high_resolution_clock::now();
 
        auto host_post_start = std::chrono::high_resolution_clock::now();
        float max_prob = 0.0f;
        int pred_cls = 0;
        for (int i = 0; i < out_features; ++i) {
            if (h_smax[i] > max_prob) {
                max_prob = h_smax[i];
                pred_cls = i;
            }
        }
        final_prediction = pred_cls;
        auto host_post_end = std::chrono::high_resolution_clock::now();

        float e2e_ms, h2d_ms, kernel_ms, d2h_ms;
        float l1_ms, l2_ms, l3_ms;
        cudaEventElapsedTime(&e2e_ms, event_e2e_start_, event_e2e_end_);
        cudaEventElapsedTime(&h2d_ms, event_h2d_start_, event_h2d_end_);
        cudaEventElapsedTime(&kernel_ms, event_l1_start_, event_kernel_end_);  // 全カーネル
        cudaEventElapsedTime(&d2h_ms, event_d2h_start_, event_d2h_end_);

        cudaEventElapsedTime(&l1_ms, event_l1_start_, event_l1_end_);
        cudaEventElapsedTime(&l2_ms, event_l2_start_, event_l2_end_);
        cudaEventElapsedTime(&l3_ms, event_l3_start_, event_l3_end_);

        double host_pre_ms = std::chrono::duration<double, std::milli>(host_pre_end - host_start).count();
        double sync_ms = std::chrono::duration<double, std::milli>(sync_end - sync_start).count();
        double host_post_ms = std::chrono::duration<double, std::milli>(host_post_end - host_post_start).count();
        
        // array保存
        e2e_times.push_back(e2e_ms);
        h2d_times.push_back(h2d_ms);
        kernel_times.push_back(kernel_ms);
        d2h_times.push_back(d2h_ms);
        host_pre_times.push_back(host_pre_ms);
        host_post_times.push_back(host_post_ms);
        sync_times.push_back(sync_ms);
        l1_kernel_times.push_back(l1_ms);
        l2_kernel_times.push_back(l2_ms);
        l3_kernel_times.push_back(l3_ms);
    }
    std::cout << " done." << std::endl;
    

    phase_stats_.host_preprocess = computeStats(host_pre_times);
    phase_stats_.gpu_transfer_h2d = computeStats(h2d_times);
    phase_stats_.gpu_kernel = computeStats(kernel_times);
    phase_stats_.gpu_transfer_d2h = computeStats(d2h_times);
    phase_stats_.host_postprocess = computeStats(host_post_times);
    phase_stats_.device_sync = computeStats(sync_times);
    phase_stats_.e2e = computeStats(e2e_times);

    layer1_stats_.h2d = phase_stats_.gpu_transfer_h2d; 
    layer1_stats_.kernel = computeStats(l1_kernel_times);
    std::vector<double> l1_totals;
    l1_totals.reserve(measure_iterations);
    for (int i = 0; i < measure_iterations; ++i) {
        l1_totals.push_back(h2d_times[i] + l1_kernel_times[i]);
    }
    layer1_stats_.total = computeStats(l1_totals);
    
    layer2_stats_.kernel = computeStats(l2_kernel_times);
    layer2_stats_.total = layer2_stats_.kernel;  // L2はカーネルのみ
    
    layer3_stats_.kernel = computeStats(l3_kernel_times);
    layer3_stats_.d2h = phase_stats_.gpu_transfer_d2h;

    std::vector<double> l3_totals;
    l3_totals.reserve(measure_iterations);
    for (int i = 0; i < measure_iterations; ++i) {
        l3_totals.push_back(l3_kernel_times[i] + d2h_times[i]);
    }
    layer3_stats_.total = computeStats(l3_totals);
    
    return final_prediction;
}
    
