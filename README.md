# V10Inference
V10Inference
C++ / CUDA によってゼロから実装した最小構成の ONNX 推論エンジンです。
学習用です　

本プロジェクトは、深層学習フレームワークを使用せず、ONNX モデルの解析から GPU 上での推論実行までを自前で実装することで、推論パイプラインの内部構造とパフォーマンス特性を深く理解することを目的としています。

また変数を適当な名前ではなく意味の名前にすること(以前から意味のない変数名が多く読みにくい問題があったので指導を受けながら変数名を意味のあるものにしています)

V10はエンジン1LR-GUE　V10エンジンのV10から来ています　

## 特徴

- CUDA: GPU を活用した高速推論

##  概要

V10Inference では以下を実装しています。

- Protocol Buffers を用いた ONNX モデル解析
- 重み抽出およびtensor整形
- CPU ↔ GPU 間の明示的なメモリ管理
- CUDA カーネルによる以下の演算実装
  - Conv2D（多チャンネル対応）
  - MaxPool
  - Fully Connected（GEMM）
  - ReLU
  - Softmax
- レイヤー単位およびフェーズ単位での詳細な性能計測

対象モデルは MNIST（Conv → Pool → Conv → Pool → FC → Softmax）です。


## 性能ベンチマーク

### 実行環境

- GPU: RTX 4050 Laptop GPU VRAM 6GB 
- CUDA: v13.0
- OS: （Windows 11 pro）
- Build: Degug 

### 計測条件

- Warmup: 20回
- 測定回数: 200回
- GPU計測: `cudaEventRecord`
- Host計測: `std::chrono`
- ベンチマークループ内でのメモリ確保やログ出力は禁止

---

## End-to-End レイテンシ（N200）

| 指標 | 値 |
|------|------|
| Mean | **0.799 ms** |
| p50 | 0.643 ms |
| p95 | 1.689 ms |
| Min | 0.529 ms |
| Max | 2.212 ms |

---

## フェーズ別内訳（平均）

| フェーズ | 時間 |
|----------|------|
| Host 前処理 | 0.088 µs |
| GPU 転送 (H2D) | 0.055 ms |
| GPU カーネル | **0.641 ms** |
| GPU 転送 (D2H) | 0.055 ms |
| 同期 | 6.19 µs |
| Host 後処理 | 0.448 µs |
---

##  レイヤー別内訳（平均）

| レイヤー | Kernel | Total |
|----------|--------|--------|
| Conv+Pool #1 | 0.124 ms | 0.180 ms |
| Conv+Pool #2 | 0.248 ms | 0.248 ms |
| FC+Softmax | 0.241 ms | 0.296 ms |


---


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
##  今後の展望(目標など)

- 一般 ONNX グラフ実行（トポロジカルソート + ディスパッチテーブル）
- テンソル shape 推論機構の実装
- メモリライフタイム解析による再利用設計
- FP16 / Tensor Core 対応
- カーネル融合による最適化
- 転送と演算の非同期オーバーラップ

---

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



## 問題があた場合

問題が発生した場合は、以下を確認してください

1. CUDA Toolkitが正しくインストールされているか 
2. 使用しているGPUがCUDA対応か
3. Visual Studioのビルド設定が正しいか


