#include <iostream>
#include <fstream>
#include <filesystem>
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
    std::cout << "Current Working Directory: " << std::filesystem::current_path() << std::endl;
    std::cout << "Redline Engine: Initializing..." << std::endl;

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

    std::cout << "Model file: " << model_path << std::endl;
    std::cout << "Image file: " << image_path << std::endl;

    if (!fileExists(model_path)) {
        std::cerr << "Error: Model file not found: " << model_path << std::endl;
        return -1;
    }

    ModelLoader model;
    if (!model.loadModel(model_path)) {
        return -1;
    }
    model.printModelInfo();

    ImageLoader imgLoader;
    std::vector<float> input_image;
    int width, height;

    if (fileExists(image_path)) {
        input_image = imgLoader.loadMNISTImage(image_path, width, height);
        if (!input_image.empty()) {
            std::cout << "Using image file: " << image_path << std::endl;
        }
    }

    if (input_image.empty()) {
        std::cout << "No image file found. Using fallback pattern..." << std::endl;
        input_image = imgLoader.createDefaultPattern();
    }
    InferenceEngine engine(model);
    int prediction = engine.run(input_image);

    std::cout << "Cleaned up GPU memory." << std::endl;
    return 0;
}