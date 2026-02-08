#include "../../include/model_loader.h"
#include <iostream>
#include <fstream>
#include <cstring>

ModelLoader::ModelLoader() {
}

ModelLoader::~ModelLoader() {
}

bool ModelLoader::loadModel(const std::string& model_path) {
    std::ifstream input(model_path, std::ios::ate | std::ios::binary);
    
    if (!input.is_open()) {
        std::cerr << "Error: Failed to open model file: " << model_path << std::endl;
        return false;
    }

    std::streamsize size = input.tellg();
    input.seekg(0, std::ios::beg);

    if (size <= 0) {
        std::cerr << "Error: File is empty or failed to read." << std::endl;
        return false;
    }

    std::cout << "Loading model (" << size << " bytes)..." << std::endl;
    
    if (!model_.ParseFromIstream(&input)) {
        std::cerr << "Error: Failed to parse ONNX model." << std::endl;
        return false;
    }

    return true;
}

std::vector<float> ModelLoader::getTensorData(const std::string& tensor_name) const {
    const auto& graph = model_.graph();
    
    for (const auto& tensor : graph.initializer()) {
        if (tensor.name() == tensor_name) {
            return extractFloatData(tensor);
        }
    }
    
    std::cerr << "Warning: Tensor '" << tensor_name << "' not found!" << std::endl;
    return std::vector<float>();
}

void ModelLoader::printModelInfo() const {
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Model IR Version: " << model_.ir_version() << std::endl;
    std::cout << "Producer Name:    " << model_.producer_name() << std::endl;
    std::cout << "Graph Name:       " << model_.graph().name() << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    std::cout << "Network Architecture (Layers):" << std::endl;
    int layer_count = 0;
    for (const auto& node : model_.graph().node()) {
        std::cout << "[" << layer_count++ << "] Operator: " << node.op_type();
        
        if (!node.name().empty()) {
            std::cout << " (" << node.name() << ")";
        }
        std::cout << std::endl;
    }
    std::cout << "------------------------------------------------" << std::endl;
}

std::vector<float> ModelLoader::extractFloatData(const onnx::TensorProto& tensor) const {
    std::vector<float> data;

    if (tensor.has_raw_data()) {
        const std::string& raw = tensor.raw_data();
        data.resize(raw.size() / sizeof(float));
        std::memcpy(data.data(), raw.data(), raw.size());
    }
    else if (tensor.float_data_size() > 0) {
        for (int i = 0; i < tensor.float_data_size(); ++i) {
            data.push_back(tensor.float_data(i));
        }
    }

    return data;
}
