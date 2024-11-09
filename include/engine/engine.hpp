#pragma once

#include <cstddef>
#include <memory>
#include <sys/types.h>
#include <vector>
#include <string>
#include <NvInfer.h>
#include "engine/logger.hpp"
#include "common/json_utils.hpp"
#include <opencv2/opencv.hpp>

namespace trt
{

    enum class Precision : int
    {
        INT8 = 8,
        FP16 = 16,
        FP32 = 32
    };

    struct EngineOptions
    {
        Precision precision = Precision::FP16;
        int32_t optBatchSize = 1;
        int32_t maxBatchSize = 1;
        int deviceIndex = 0;
    };

    struct EngineConfig : public JsonConfig
    {
        std::string modelPath{};
        int batchSize = 1;
        Precision precision = Precision::FP16;

        std::shared_ptr<const JsonConfig> clone() const override { return std::make_shared<EngineConfig>(*this); }

        void loadFromJson(const nlohmann::json &data) override {
            if (data.contains("model_path"))
                modelPath = data["model_path"].get<std::string>();
            if (data.contains("batch_size"))
                batchSize = data["batch_size"].get<int>();
            if (data.contains("precision"))
                precision = static_cast<Precision>(data["precision"].get<int>());
        }
    };

    class Engine
    {
    public:
        Engine(const EngineOptions &options);
        ~Engine();
        // Clear memory
        void clearBuffers();
        // Load and prepare engine for inference
        bool loadNetwork(const std::string &engineModelPath);
        // Load inputs to CUDA memory
        bool prepareInputs(const std::vector<std::vector<cv::Mat>> &inputs, cudaStream_t &inferenceCudaStream, const int32_t batchSize);
        // Copy the outputs back to CPU
        bool prepareOutputs(std::vector<std::vector<std::vector<float>>> &outputs, cudaStream_t &inferenceCudaStream, const int32_t batchSize);

        // Run inference
        // Input format: [input][batch][cv::Mat]
        // Output format: [batch][output][feature_vector]
        bool runInference(const cv::Mat image, std::vector<float> &featureVector);
        bool runInference(const std::vector<cv::Mat> &inputBatch, std::vector<std::vector<float>> &outputBatch);
        bool runInference(const std::vector<std::vector<cv::Mat>> &inputs, std::vector<std::vector<std::vector<float>>> &outputs);

        [[nodiscard]] const std::vector<nvinfer1::Dims3> &getInputDims() const { return m_inputDims; };
        [[nodiscard]] const std::vector<nvinfer1::Dims> &getOutputDims() const { return m_outputDims; };

    private:
        // Holds pointer to the input and output GPU buffers
        std::vector<void *> m_buffers{};
        std::vector<uint32_t> m_outputLengths{};
        std::vector<nvinfer1::Dims3> m_inputDims{};
        std::vector<nvinfer1::Dims> m_outputDims{};
        std::vector<std::string> m_IOTensorNames{};

        std::unique_ptr<nvinfer1::IRuntime> m_runtime = nullptr;
        std::unique_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;
        std::unique_ptr<nvinfer1::IExecutionContext> m_context = nullptr;

        NvLogger m_logger{};
        const EngineOptions m_options;
    };

    bool loadEngine(Engine &engine, const std::string &engineModelPath);
    void setEngineOptions(EngineOptions &options, int batchSize, Precision precision);

} // namespace trt