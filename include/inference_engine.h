#pragma once

#include <vector>
#include "model_loader.h"

// 推論エンジンクラス
// CNNの各層を実行し、最終的な分類結果を返す
class InferenceEngine {
public:
    InferenceEngine(const ModelLoader& model);
    ~InferenceEngine();

    // GPUメモリの事前確保・モデル重みの転送
    bool allocateGPU();

    // 推論を実行
    int run(const std::vector<float>& input_image);

    // 各層の処理時間を取得 (ms)
    double getLayer1Time() const { return layer1_time_ms; }
    double getLayer2Time() const { return layer2_time_ms; }
    double getLayer3Time() const { return layer3_time_ms; }

private:
    const ModelLoader& model_;
    bool gpu_allocated_ = false;

    // タイミング情報 (ms)
    double layer1_time_ms = 0.0;
    double layer2_time_ms = 0.0;
    double layer3_time_ms = 0.0;

    // --- 事前確保GPU メモリ ---
    // 入力バッファ
    float* d_input_ = nullptr;        // 28*28

    // Layer 1 重み・バイアス・中間バッファ
    float* d_l1_weights_ = nullptr;   // 8*1*5*5
    float* d_l1_bias_ = nullptr;      // 8
    float* d_l1_conv_out_ = nullptr;  // 8*24*24
    float* d_l1_pool_out_ = nullptr;  // 8*12*12

    // Layer 2 重み・バイアス・中間バッファ
    float* d_l2_weights_ = nullptr;   // 16*8*5*5
    float* d_l2_bias_ = nullptr;      // 16
    float* d_l2_conv_out_ = nullptr;  // 16*8*8
    float* d_l2_pool_out_ = nullptr;  // 16*4*4

    // Layer 3 重み・バイアス・出力バッファ
    float* d_l3_weights_ = nullptr;   // 10*256
    float* d_l3_bias_ = nullptr;      // 10
    float* d_l3_fc_out_ = nullptr;    // 10
    float* d_l3_smax_out_ = nullptr;  // 10

    // 第1層: Conv2D + Bias + ReLU + MaxPool
    void runLayer1(const std::vector<float>& input);

    // 第2層: マルチチャンネルConv2D + Bias + ReLU + MaxPool
    void runLayer2();

    // 第3層: 全結合層 + Softmax
    int runLayer3();

    // GPU メモリ解放
    void freeGPU();

    // ヘルパー関数
    void printFilterResults(const std::vector<float>& data, int num_filters, 
                           int pix_per_filt, const std::string& label);
};
