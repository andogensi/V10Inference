#include "../../include/inference_engine.h"
#include <iostream>
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
}

int InferenceEngine::run(const std::vector<float>& input_image) {
    std::cout << "================================================" << std::endl;
    std::cout << "Starting Inference..." << std::endl;
    std::cout << "================================================" << std::endl;

    float* d_l1_out = nullptr;
    float* d_l2_out = nullptr;
 // 第1層
    runLayer1(input_image, &d_l1_out);

    // 第2層
    runLayer2(d_l1_out, &d_l2_out);
  // 第3層
    int prediction = runLayer3(d_l2_out);

    //解放
    cudaFree(d_l1_out);
    cudaFree(d_l2_out);

    return prediction;
}

void InferenceEngine::runLayer1(const std::vector<float>& input, float** d_pout) {
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Layer 1: Conv2D (8 filters) + Bias + ReLU + MaxPool" << std::endl;

    const int input_w = 28;
    const int input_h = 28;
    const int kernel_size = 5;
    const int num_filters = 8;
    const int output_w = input_w - kernel_size + 1; // 24
    const int output_h = input_h - kernel_size + 1; // 24
    const int ch_pix = output_w * output_h; // 576
    std::vector<float> h_weights = model_.getTensorData("Parameter5");
    std::vector<float> h_bias = model_.getTensorData("Parameter6");

    if (h_weights.empty() || h_bias.empty()) {
        std::cerr << "Error: Layer 1 parameters not found!" << std::endl;
        return;
    }

    //メモリ
    float *d_input, *d_weights, *d_bias, *d_output;
    cudaMalloc(&d_input, input.size() * sizeof(float));
    cudaMalloc(&d_weights, h_weights.size() * sizeof(float));
    cudaMalloc(&d_bias, h_bias.size() * sizeof(float));
    cudaMalloc(&d_output, num_filters * ch_pix * sizeof(float));

    //転送
    cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights.data(), h_weights.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias.data(), h_bias.size() * sizeof(float), cudaMemcpyHostToDevice);

    //(8フィルタ)
    int filt_w_cnt = kernel_size * kernel_size;
    for (int i = 0; i < num_filters; ++i) {
        float* w_ptr = d_weights + (i * filt_w_cnt);
        float* out_ptr = d_output + (i * ch_pix);
        launchConv2D(d_input, w_ptr, out_ptr, 
                     input_w, input_h, kernel_size, output_w, output_h);
    }

    // Bias + ReLU
    launchAddBiasRelu(d_output, d_bias, ch_pix, num_filters);

#ifdef DEBUG_PRINT
    std::vector<float> h_output(num_filters * ch_pix);
    cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
    printFilterResults(h_output, num_filters, ch_pix, "After Conv + ReLU (24x24)");
#endif

    const int pw = output_w / 2; // 12
    const int ph = output_h / 2; // 12
    const int p_pix = pw * ph; // 144

    cudaMalloc(d_pout, num_filters * p_pix * sizeof(float));

    for (int i = 0; i < num_filters; ++i) {
        float* in_ptr = d_output + (i * ch_pix);
        float* out_ptr = *d_pout + (i * p_pix);
        launchMaxPool(in_ptr, out_ptr, 
                      output_w, output_h, pw, ph);
    }

#ifdef DEBUG_PRINT
    // 結果確認
    std::vector<float> h_pout(num_filters * p_pix);
    cudaMemcpy(h_pout.data(), *d_pout, h_pout.size() * sizeof(float), cudaMemcpyDeviceToHost);
    printFilterResults(h_pout, num_filters, p_pix, "After MaxPool (12x12)");
#endif

    // クリーンアップ
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_bias);
    cudaFree(d_output);

    std::cout << "Layer 1 kansei!" << std::endl;
}

void InferenceEngine::runLayer2(float* d_l1_out, float** d_pout) {
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Layer 2: Conv2D (16 filters) + Bias + ReLU + MaxPool" << std::endl;

    const int in_w = 12;
    const int in_h = 12;
    const int in_channels = 8;
    const int kernel_size = 5;
    const int out_channels = 16;
    const int out_w = in_w - kernel_size + 1; // 8
    const int out_h = in_h - kernel_size + 1; // 8
    const int out_pix = out_w * out_h; // 64

    // 重みとバイアスを取得
    std::vector<float> h_weights = model_.getTensorData("Parameter87");
    std::vector<float> h_bias = model_.getTensorData("Parameter88");

    if (h_weights.empty() || h_bias.empty()) {
        std::cerr << "Error: Layer 2 parameters not found!" << std::endl;
        return;
    }

    // GPUメモリ確保
    float *d_weights, *d_bias, *d_output;
    cudaMalloc(&d_weights, h_weights.size() * sizeof(float));
    cudaMalloc(&d_bias, h_bias.size() * sizeof(float));
    cudaMalloc(&d_output, out_channels * out_pix * sizeof(float));

    // データ転送
    cudaMemcpy(d_weights, h_weights.data(), h_weights.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias.data(), h_bias.size() * sizeof(float), cudaMemcpyHostToDevice);

    // マルチチャンネルConv2D
    launchConv2DMultiChannel(d_l1_out, d_weights, d_output,
                             in_w, in_h, in_channels, kernel_size, out_w, out_h, out_channels);

    launchAddBiasRelu(d_output, d_bias, out_pix, out_channels);

#ifdef DEBUG_PRINT
    std::vector<float> h_output(out_channels * out_pix);
    cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
    printFilterResults(h_output, out_channels, out_pix, "After Conv + ReLU (8x8)");
#endif

    const int pw = out_w / 2; // 4
    const int ph = out_h / 2; // 4
    const int p_pix = pw * ph; // 16

    cudaMalloc(d_pout, out_channels * p_pix * sizeof(float));

    launchMaxPoolMultiChannel(d_output, *d_pout, out_w, out_h, 
                              pw, ph, out_channels);
    
#ifdef DEBUG_PRINT
    std::vector<float> h_pout(out_channels * p_pix);
    cudaMemcpy(h_pout.data(), *d_pout, h_pout.size() * sizeof(float), cudaMemcpyDeviceToHost);
    printFilterResults(h_pout, out_channels, p_pix, "After MaxPool (4x4)");
#endif

    cudaFree(d_weights);
    cudaFree(d_bias);
    cudaFree(d_output);

    std::cout << "Layer 2 Complete!" << std::endl;
}

int InferenceEngine::runLayer3(float* d_l2_out) {
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Layer 3: Fully Connected (256 -> 10) + Softmax" << std::endl;

    const int in_features = 16 * 16; 
    const int out_features = 10;

    std::vector<float> h_weights = model_.getTensorData("Parameter193");
    std::vector<float> h_bias = model_.getTensorData("Parameter194");

    if (h_weights.empty() || h_bias.empty()) {
        std::cerr << "Error: Layer 3 parameters not found!" << std::endl;
        return -1;
    }

    float *d_weights, *d_bias, *d_output, *d_smax;
    cudaMalloc(&d_weights, h_weights.size() * sizeof(float));
    cudaMalloc(&d_bias, h_bias.size() * sizeof(float));
    cudaMalloc(&d_output, out_features * sizeof(float));
    cudaMalloc(&d_smax, out_features * sizeof(float));

    cudaMemcpy(d_weights, h_weights.data(), h_weights.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias.data(), h_bias.size() * sizeof(float), cudaMemcpyHostToDevice);

    launchFullyConnected(d_l2_out, d_weights, d_bias, d_output, in_features, out_features);

#ifdef DEBUG_PRINT
    std::vector<float> h_output(out_features);
    cudaMemcpy(h_output.data(), d_output, out_features * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "--- FC Output (Raw Logits) ---" << std::endl;
    for (int i = 0; i < out_features; ++i) {
        std::cout << "Class [" << i << "]: " << h_output[i] << std::endl;
    }
#endif

    launchSoftmax(d_output, d_smax, out_features);

    std::vector<float> h_smax(out_features);
    cudaMemcpy(h_smax.data(), d_smax, out_features * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "================================================" << std::endl;
    std::cout << "Inference Results (Probabilities)" << std::endl;
    std::cout << "================================================" << std::endl;
    
    float max_prob = 0.0f;
    int pred_cls = 0;
    
    for (int i = 0; i < out_features; ++i) {
        std::cout << "Digit " << i << ": " << (h_smax[i] * 100.0f) << "%" << std::endl;
        if (h_smax[i] > max_prob) {
            max_prob = h_smax[i];
            pred_cls = i;
        }
    }

    std::cout << "================================================" << std::endl;
    std::cout << "PREDICTION: " << pred_cls << " (Confidence: " << (max_prob * 100.0f) << "%)" << std::endl;
    std::cout << "================================================" << std::endl;
    //術式解放
    cudaFree(d_weights);
    cudaFree(d_bias);
    cudaFree(d_output);
    cudaFree(d_smax);

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
