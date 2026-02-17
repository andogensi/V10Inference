#pragma once

#include <string>
#include <vector>

class ImageLoader {
public:
    ImageLoader();
    ~ImageLoader();

    // 画像ファイルを読み込む　正規化も
    std::vector<float> loadMNISTImage(const std::string& path, int& width, int& height);
    std::vector<float> createDefaultPattern();

private:
    static const int MNIST_WIDTH = 28;
    static const int MNIST_HEIGHT = 28;
};
