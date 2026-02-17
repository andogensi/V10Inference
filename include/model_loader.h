#pragma once

#include <string>
#include <vector>
#include "../third_party/onnx/onnx.pb.h"

// ONNXモデルのローダ
class ModelLoader {
public:
    ModelLoader();
    ~ModelLoader();

    // モデルファイルを読みこみ
    bool loadModel(const std::string& model_path);
    const onnx::ModelProto& getModel() const { return model_; }
    const onnx::GraphProto& getGraph() const { return model_.graph(); }


    std::vector<float> getTensorData(const std::string& tensor_name) const;

    void printModelInfo() const;

private:
    onnx::ModelProto model_;

    std::vector<float> extractFloatData(const onnx::TensorProto& tensor) const;
};
