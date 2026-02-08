#pragma once

#include <string>
#include <vector>
#include "../third_party/onnx/onnx.pb.h"

// ONNXモデルのローダークラス
class ModelLoader {
public:
    ModelLoader();
    ~ModelLoader();

    // モデルファイルを読み込む
    bool loadModel(const std::string& model_path);

    // モデル情報を取得
    const onnx::ModelProto& getModel() const { return model_; }
    const onnx::GraphProto& getGraph() const { return model_.graph(); }

    // 指定された名前のテンsorデータを取得
    std::vector<float> getTensorData(const std::string& tensor_name) const;

    // モデル情報を表示
    void printModelInfo() const;

private:
    onnx::ModelProto model_;
    
    // TensorProtoからfloat配列への変換
    std::vector<float> extractFloatData(const onnx::TensorProto& tensor) const;
};
