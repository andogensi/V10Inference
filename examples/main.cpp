#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include "../include/model_loader.h"
#include "../include/image_loader.h"
#include "../include/inference_engine.h"

bool fileExists(const std::string& filename) {
    std::ifstream f(filename.c_str());
    return f.good();
}

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]" << std::endl;
    std::cout << "\nOptions:" << std::endl;
    std::cout << "  -i, --image <path>    Input image file path (default: test_digit.png)" << std::endl;
    std::cout << "  -m, --model <path>    ONNX model file path (default: mnist-8.onnx)" << std::endl;
    std::cout << "  -h, --help            Show this help message" << std::endl;
    std::cout << "\nExample:" << std::endl;
    std::cout << "  " << program_name << " -i my_digit.png -m mnist-8.onnx" << std::endl;
    std::cout << "  " << program_name << " --image my_digit.png" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "V10: Initializing..." << std::endl;

    // デフォルト
    std::string model_path = "mnist-8.onnx";
    std::string image_path = "test_digit.png";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        }
        else if (arg == "-i" || arg == "--image") {
            if (i + 1 < argc) {
                image_path = argv[++i];
            } else {
                std::cerr << "Error: --image requires an argument" << std::endl;
                printUsage(argv[0]);
                return -1;
            }
        }
        else if (arg == "-m" || arg == "--model") {
            if (i + 1 < argc) {
                model_path = argv[++i];
            } else {
                std::cerr << "Error: --model requires an argument" << std::endl;
                printUsage(argv[0]);
                return -1;
            }
        }
        else {
            std::cerr << "Error: Unknown option: " << arg << std::endl;
            printUsage(argv[0]);
            return -1;
        }
    }

    if (!fileExists(model_path)) {
        std::cerr << "Error: Model file not found: " << model_path << std::endl;
        return -1;
    }

    ModelLoader model;
    if (!model.loadModel(model_path)) {
        return -1;
    }

    ImageLoader imgLoader;
    std::vector<float> input_image;
    int width, height;

    if (fileExists(image_path)) {
        input_image = imgLoader.loadMNISTImage(image_path, width, height);
    }

    if (input_image.empty()) {
        input_image = imgLoader.createDefaultPattern();
    }

    InferenceEngine engine(model);

    std::cout << "\n[GPU Setup]" << std::endl;
    auto setup_start = std::chrono::high_resolution_clock::now();
    if (!engine.allocateGPU()) {
        std::cerr << "Error: GPU allocation failed!" << std::endl;
        return -1;
    }
    auto setup_end = std::chrono::high_resolution_clock::now();
    auto setup_ms = std::chrono::duration<double, std::milli>(setup_end - setup_start).count();
    std::cout << "GPU setup time: " << setup_ms << " ms" << std::endl;
    
    std::cout << "\n[Warmup run]" << std::endl;
    engine.run(input_image);
    std::cout << "\n[Actual measurement]" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    int prediction = engine.run(input_image);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    std::cout << "\n=== Resulet ===" << std::endl;
    std::cout << "inference: " << prediction << std::endl;
    std::cout << "time: " << duration_ms << " ms (" << duration_us / 1000.0 << " ms)" << std::endl;
    std::cout << "\n=== Layer Details ===" << std::endl;
    std::cout << "Layer 1 (Conv+Pool): " << engine.getLayer1Time() << " ms" << std::endl;
    std::cout << "Layer 2 (Conv+Pool): " << engine.getLayer2Time() << " ms" << std::endl;
    std::cout << "Layer 3 (FC+Softmax): " << engine.getLayer3Time() << " ms" << std::endl;
    std::cout << "Total (sum): " << (engine.getLayer1Time() + engine.getLayer2Time() + engine.getLayer3Time()) << " ms" << std::endl;

    return 0;
}