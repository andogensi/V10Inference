#pragma once

#include <vector>
#include "model_loader.h"

// 推論エンジンクラス
// CNNの各層を実行し、最終的な分類結果を返す
class InferenceEngine {
public:
    InferenceEngine(const ModelLoader& model);
    ~InferenceEngine();

    // 推論を実行
    int run(const std::vector<float>& input_image);

private:
    const ModelLoader& model_;

    // 第1層: Conv2D + Bias + ReLU + MaxPool
    void runLayer1(const std::vector<float>& input, float** d_pout);

    // 第2層: マルチチャンネルConv2D + Bias + ReLU + MaxPool
    void runLayer2(float* d_l1_out, float** d_pout);

    // 第3層: 全結合層 + Softmax
    int runLayer3(float* d_l2_out);

    // ヘルパー関数
    void printFilterResults(const std::vector<float>& data, int num_filters, 
                           int pix_per_filt, const std::string& label);
};
