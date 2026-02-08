#pragma once

#include <string>
#include <vector>

// MNIST画像の読み込みを行うクラス
class ImageLoader {
public:
    ImageLoader();
    ~ImageLoader();

    // 画像ファイルを読み込む (28x28、グレースケール、0-1正規化)
    std::vector<float> loadMNISTImage(const std::string& path, int& width, int& height);

    // フォールバック: デフォルトの「1」パターンを生成
    std::vector<float> createDefaultPattern();

private:
    static const int MNIST_WIDTH = 28;
    static const int MNIST_HEIGHT = 28;
};
