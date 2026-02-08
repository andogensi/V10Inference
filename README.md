# V10Inference

V10Inference は、ONNX形式の機械学習モデルを使用した高速推論エンジンです。CUDA対応のGPUアクセラレーションにより、効率的なニューラルネットワーク推論を実現します。

## 特徴

- 🚀 **CUDA アクセラレーション**: GPU を活用した高速推論
- 📦 **ONNX サポート**: 標準的なONNX形式のモデルに対応
- 🎯 **画像分類**: MNIST などの画像認識タスクに対応
- 🔧 **モジュラー設計**: 拡張しやすいアーキテクチャ

## ディレクトリ構成

```
V10Inference/
├── include/               # 公開ヘッダーファイル
│   ├── inference_engine.h
│   ├── image_loader.h
│   └── model_loader.h
├── src/                   # 実装ファイル
│   ├── core/             # コア機能
│   │   └── inference_engine.cpp
│   ├── loaders/          # データ読み込み
│   │   ├── image_loader.cpp
│   │   └── model_loader.cpp
│   └── cuda/             # CUDAカーネル
│       └── kernels.cu
├── third_party/          # サードパーティライブラリ
│   └── onnx/
│       ├── onnx.pb.h
│       └── onnx.pb.cc
└── examples/             # サンプルコード
    └── main.cpp
```

## 必要要件

### ソフトウェア
- Visual Studio 2022 (またはそれ以降)
- CUDA Toolkit 13.0 (またはそれ以降)
- C++17 対応コンパイラ

### ハードウェア
- CUDA対応 NVIDIA GPU

## ビルド方法

### Visual Studio を使用する場合

1. `V10Inference.sln` を Visual Studio で開く
2. ビルド構成を選択 (Debug または Release)
3. プラットフォームを x64 に設定
4. ビルド → ソリューションのビルド

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
| `-i, --image <path>` | 入力画像ファイルのパス | `test_digit.png` |
| `-m, --model <path>` | ONNXモデルファイルのパス | `mnist-8.onnx` |
| `-h, --help` | ヘルプメッセージを表示 | - |

### プログラムでの使用例

```cpp
#include "model_loader.h"
#include "image_loader.h"
#include "inference_engine.h"

int main() {
    // モデルの読み込み
    ModelLoader model;
    model.loadModel("mnist-8.onnx");
    
    // 画像の読み込み
    ImageLoader imgLoader;
    int width, height;
    auto image = imgLoader.loadMNISTImage("test.png", width, height);
    
    // 推論の実行
    InferenceEngine engine(model);
    int prediction = engine.run(image);
    
    std::cout << "予測結果: " << prediction << std::endl;
    return 0;
}
```

## API リファレンス

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

ニューラルネットワークの推論を実行するメインエンジン。

```cpp
class InferenceEngine {
public:
    InferenceEngine(const ModelLoader& model);
    int run(const std::vector<float>& input_image);
};
```

## ライセンス

このプロジェクトのライセンス情報については、プロジェクトオーナーにお問い合わせください。

## 貢献

バグ報告や機能要望は、GitHubのIssuesセクションにお願いします。

## サポート

問題が発生した場合は、以下を確認してください:

1. CUDA Toolkitが正しくインストールされているか
2. 使用しているGPUがCUDA対応か
3. Visual Studioのビルド設定が正しい]

---

**Made with ❤️ for High-Performance AI Inference**
