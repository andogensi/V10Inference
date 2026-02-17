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
    std::cout << " " << program_name << " -i my_digit.png -m mnist-8.onnx" << std::endl;
    std::cout << " " << program_name << " --image my_digit.png" << std::endl;
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
                std::cerr << "Error: --image t" << std::endl;
                printUsage(argv[0]);
                return -1;
            }
        }
        else if (arg == "-m" || arg == "--model") {
            if (i + 1 < argc) {
                model_path = argv[++i];
            } else {
                std::cerr << "Error: --modelt" << std::endl;
                printUsage(argv[0]);
                return -1;
            }
        }
        else {
            std::cerr << "Error: Unknow " << arg << std::endl;
            printUsage(argv[0]);
            return -1;
        }
    }

    if (!fileExists(model_path)) {
        std::cerr << "Error: Model file not " << model_path << std::endl;
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
        std::cerr << "Error: GPU allocation failed" << std::endl;
        return -1;
    }
    auto setup_end = std::chrono::high_resolution_clock::now();
    auto setup_ms = std::chrono::duration<double, std::milli>(setup_end - setup_start).count();
    std::cout << "GPU setup time: " << setup_ms << " ms" << std::endl;
    constexpr int WARMUP_ITERATIONS = 20;
	constexpr int MEASURE_ITERATIONS = 200;//100から200に
    
    std::cout << "\n[Statistical Benchmark - Portfolio Quality]" << std::endl;
    std::cout << "Configuration: Warmup=" << WARMUP_ITERATIONS 
              << ", Measure=" << MEASURE_ITERATIONS << std::endl;
    
    int prediction = engine.runWithStats(input_image, WARMUP_ITERATIONS, MEASURE_ITERATIONS);
    
    std::cout << "\n=== Final Prediction ===" << std::endl;
    std::cout << "Predicted digit: " << prediction << std::endl;
    engine.printTimingStats();

    return 0;
}