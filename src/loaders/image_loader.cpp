#include "../../include/image_loader.h"
#include <iostream>

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"
#endif

ImageLoader::ImageLoader() {
}

ImageLoader::~ImageLoader() {
}

std::vector<float> ImageLoader::loadMNISTImage(const std::string& path, int& width, int& height) {
    std::vector<float> result;
    int channels;
    
    unsigned char* data = stbi_load(path.c_str(), &width, &height, &channels, 1);
    if (!data) {
        std::cerr << "Error: Failed to load image: " << path << std::endl;
        std::cerr << "stbi_error: " << stbi_failure_reason() << std::endl;
        return result;
    }
    
    std::cout << "Loaded image: " << path << " (" << width << "x" << height 
              << ", channels=" << channels << ")" << std::endl;
    
    if (width != MNIST_WIDTH || height != MNIST_HEIGHT) {
        std::cerr << "Warning: Image is not 28x28. Expected 28x28 for MNIST." << std::endl;
    }
    
    // 正規化 (0-255 0.0-1.0)
    result.resize(width * height);
    for (int i = 0; i < width * height; ++i) {
        result[i] = static_cast<float>(data[i]) / 255.0f;
    }
    
    stbi_image_free(data);
    return result;
}

std::vector<float> ImageLoader::createDefaultPattern() {
    std::vector<float> h_input(MNIST_WIDTH * MNIST_HEIGHT, 0.0f);
    
    std::cout << "Creating default pattern (digit '1')..." << std::endl;
    
    // 「1」のパターン: 中央付近に縦線を描画
    for (int y = 4; y < 24; ++y) {
        for (int x = 13; x <= 15; ++x) {
            h_input[y * 28 + x] = 1.0f;
        }
    }
    for (int x = 11; x <= 15; ++x) {
        h_input[4 * 28 + x] = 1.0f;
        h_input[5 * 28 + x] = 1.0f;
    }
    for (int x = 10; x <= 18; ++x) {
        h_input[22 * 28 + x] = 1.0f;
        h_input[23 * 28 + x] = 1.0f;
    }
    
    return h_input;
}
