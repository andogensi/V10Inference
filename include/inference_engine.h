#pragma once

#include <vector>
#include <cuda_runtime.h>
#include "model_loader.h"


struct TimingStats {
    double mean_ms = 0.0;
    double std_dev_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double p50_ms = 0.0;    // 中央値
    double p95_ms = 0.0;    // 95
    int num_samples = 0;
};
struct PhaseTiming {
    TimingStats host_preprocess;    
    TimingStats gpu_transfer_h2d;   
    TimingStats gpu_kernel;         
    TimingStats gpu_transfer_d2h;   
    TimingStats host_postprocess;   
    TimingStats device_sync;        
    TimingStats e2e;                
};


struct LayerTiming {
    TimingStats h2d;        
    TimingStats kernel;     
    TimingStats d2h;        
    TimingStats total;     
};
//推論エンジンクラス
class InferenceEngine {
public:
    InferenceEngine(const ModelLoader& model);
    ~InferenceEngine();

    bool allocateGPU();

    int run(const std::vector<float>& input_image);
    int runSilent(const std::vector<float>& input_image);
    int runWithStats(const std::vector<float>& input_image, 
                     int warmup_iterations = 20, 
                     int measure_iterations = 200);
    double getLayer1Time() const { return layer1_time_ms; }
    double getLayer2Time() const { return layer2_time_ms; }
    double getLayer3Time() const { return layer3_time_ms; }

    const LayerTiming& getLayer1Stats() const { return layer1_stats_; }
    const LayerTiming& getLayer2Stats() const { return layer2_stats_; }
    const LayerTiming& getLayer3Stats() const { return layer3_stats_; }
    
    const PhaseTiming& getPhaseStats() const { return phase_stats_; }
    
    // タイミング統計
    void printTimingStats() const;

private:
    const ModelLoader& model_;
    bool gpu_allocated_ = false;

    double layer1_time_ms = 0.0;
    double layer2_time_ms = 0.0;
    double layer3_time_ms = 0.0;
    double last_l1_h2d_ = 0.0;
    double last_l1_kernel_ = 0.0;
    double last_l1_d2h_ = 0.0;
    double last_l2_h2d_ = 0.0;
    double last_l2_kernel_ = 0.0;
    double last_l2_d2h_ = 0.0;
    double last_l3_h2d_ = 0.0;
    double last_l3_kernel_ = 0.0;
    double last_l3_d2h_ = 0.0;
    LayerTiming layer1_stats_;
    LayerTiming layer2_stats_;
    LayerTiming layer3_stats_;
    PhaseTiming phase_stats_;
    
    //Stream 
    cudaStream_t stream_ = nullptr;
    cudaEvent_t event_start_, event_end_;
    cudaEvent_t event_h2d_start_, event_h2d_end_;
    cudaEvent_t event_kernel_start_, event_kernel_end_;
    cudaEvent_t event_d2h_start_, event_d2h_end_;
    cudaEvent_t event_e2e_start_, event_e2e_end_;
    //カーネル計測用
    cudaEvent_t event_l1_start_, event_l1_end_;
    cudaEvent_t event_l2_start_, event_l2_end_;
    cudaEvent_t event_l3_start_, event_l3_end_;
    bool events_created_ = false;
    float* d_input_ = nullptr;        // 28*28

    // Layer1
    float* d_l1_weights_ = nullptr;   // 8*1*5*5
    float* d_l1_bias_ = nullptr;      // 8
    float* d_l1_conv_out_ = nullptr;  // 8*24*24
    float* d_l1_pool_out_ = nullptr;  // 8*12*12

    // Layer2
    float* d_l2_weights_ = nullptr;   // 16*8*5*5
    float* d_l2_bias_ = nullptr;      // 16
    float* d_l2_conv_out_ = nullptr;  // 16*8*8
    float* d_l2_pool_out_ = nullptr;  // 16*4*4

    // Layer3
    float* d_l3_weights_ = nullptr;   // 10*256
    float* d_l3_bias_ = nullptr;      // 10
    float* d_l3_fc_out_ = nullptr;    // 10
    float* d_l3_smax_out_ = nullptr;  // 10


    void runLayer1Silent(const std::vector<float>& input);
    void runLayer2Silent();

    int runLayer3Silent();
    void runLayer1(const std::vector<float>& input);
    void runLayer2();
    int runLayer3();
    void freeGPU();
    void createTimingEvents();
    void destroyTimingEvents();
    
    // p50/p95
    static TimingStats computeStats(const std::vector<double>& times);
    static double computePercentile(std::vector<double> sorted_times, double percentile);
    void printFilterResults(const std::vector<float>& data, int num_filters, 
                           int pix_per_filt, const std::string& label);
};
