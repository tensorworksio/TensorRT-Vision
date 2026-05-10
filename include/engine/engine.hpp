#pragma once

#include <cstddef>
#include <memory>
#include <sys/types.h>
#include <vector>
#include <string>
#include <NvInfer.h>
#include "engine/logger.hpp"
#include <opencv2/opencv.hpp>

namespace trt
{

    enum class Precision { INT8, FP16, FP32 };

    struct EngineOptions
    {
        Precision precision = Precision::FP16;
        int32_t optBatchSize = 1;
        int32_t maxBatchSize = 1;
        int deviceIndex = 0;
    };

    struct EngineConfig
    {
        std::string model_path{};
        int batch_size = 1;
        Precision precision = Precision::FP16;
    };

    class Engine
    {
    public:
        Engine(const EngineOptions &options);
        ~Engine();
        void clearBuffers();
        void loadNetwork(const std::string &engineModelPath);
        void prepareInputs(const std::vector<std::vector<cv::Mat>> &inputs, cudaStream_t &inferenceCudaStream, const int32_t batchSize);
        void prepareOutputs(std::vector<std::vector<std::vector<float>>> &outputs, cudaStream_t &inferenceCudaStream, const int32_t batchSize);

        void runInference(const cv::Mat &image, std::vector<float> &featureVector);
        void runInference(const cv::Mat &image, std::vector<std::vector<float>> &outputs);
        void runInference(const std::vector<cv::Mat> &inputBatch, std::vector<std::vector<float>> &outputBatch);
        void runInference(const std::vector<cv::Mat> &inputBatch, std::vector<std::vector<std::vector<float>>> &outputBatch);
        void runInference(const std::vector<std::vector<cv::Mat>> &inputs, std::vector<std::vector<std::vector<float>>> &outputs);

        [[nodiscard]] const EngineOptions &getOptions() const { return m_options; };
        [[nodiscard]] const std::vector<nvinfer1::Dims3> &getInputDims() const { return m_inputDims; };
        [[nodiscard]] const std::vector<nvinfer1::Dims> &getOutputDims() const { return m_outputDims; };

    private:
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

    void loadEngine(Engine &engine, const std::string &engineModelPath);
    void setEngineOptions(EngineOptions &options, int batchSize, Precision precision);

} // namespace trt
