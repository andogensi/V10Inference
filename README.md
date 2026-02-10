# V10Inference

AIの仕組みについての理解を深めたいという思いで個人開発しています 
V10Inference は、ONNX形式の機械学習モデル用のAI推論エンジンです
V10はエンジン1LR-GUE　V10エンジンのV10から来ています　

## 特徴

- **CUDA: GPU を活用した高速推論


## ディレクトリ構成

```
V10Inference/
├── include/               # 公開ヘッダーファイル
│   ├── inference_engine.h
│   ├── image_loader.h
│   └── model_loader.h
├── src/                   # 実装ファイル
│   ├── core/             # コア
│   │   └── inference_engine.cpp
│   ├── loaders/          # 読み込み
│   │   ├── image_loader.cpp
│   │   └── model_loader.cpp
│   └── cuda/             # CUDAカーネル
│       └── kernels.cu
├── third_party/          # サードパーティライブラリ
│   └── onnx/
│       ├── onnx.pb.h
│       └── onnx.pb.cc
└── examples/             # サンプル
    └── main.cpp
```

## 要件

### ソフトウェア
- Visual Studio 2022
- CUDA Toolkit 13.0
- C++17 対応コンパイラ(MSVC v143)

### ハードウェア
- CUDA NVIDIA GPU

## 使用方法

### 基本的な使い方

```bash
# デフォルト設定で実行
V10Inference.exe

# 画像ファイルを指定
V10Inference.exe -i my_digit.png

# モデルとファイルの両方を指定
V10Inference.exe -i my_digit.png -m mnist-8.onnx
```

### コマンドライン オプション

| オプション | 説明 | デフォルト値 |
|-----------|------|-------------|
| `-i, --image path` | 入力画像ファイルのパス | `test_digit.png` |
| `-m, --model path` | ONNXのパス | `mnist-8.onnx` |

## API 

### ModelLoader

ONNXモデルファイルを読み込むためのクラス。

```cpp
class ModelLoader {
public:
    bool loadModel(const std::string& model_path);
    std::vector<float> getTensorData(const std::string& tensor_name) const;
    void printModelInfo() const;
};
```

### ImageLoader

画像ファイルを読み込み、推論用のデータに変換するクラス。

```cpp
class ImageLoader {
public:
    std::vector<float> loadMNISTImage(const std::string& path, int& width, int& height);
    std::vector<float> createDefaultPattern();
};
```

### InferenceEngine

ニューラルネットワークの推論を実行するメインクラス

```cpp
class InferenceEngine {
public:
    InferenceEngine(const ModelLoader& model);
    int run(const std::vector<float>& input_image);
};
```

## 問題があた場合

問題が発生した場合は、以下を確認してください

1. CUDA Toolkitが正しくインストールされているか 
2. 使用しているGPUがCUDA対応か
3. Visual Studioのビルド設定が正しいか


