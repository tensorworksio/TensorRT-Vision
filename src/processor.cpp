#include "engine/processor.hpp"

namespace trt
{

    ModelProcessor::ModelProcessor(const EngineConfig &config) : m_config(config.clone())
    {
        // Engine options
        NvLogger logger;
        EngineOptions options;
        setEngineOptions(options, config.batchSize, config.precision);

        // Load engine
        m_trtEngine = std::make_unique<Engine>(options, logger);
        loadEngine(*m_trtEngine, config.engineModelPath);
    }

    bool ModelProcessor::process(const cv::Mat &image, std::vector<Detection> &detections)
    {
        bool success;
        cv::Mat processedImage;
        std::vector<float> featureVector;

        success = preprocess(image, processedImage);
        if (!success)
        {
            std::runtime_error("Model preprocessing failed");
        }
        success = m_trtEngine->runInference(processedImage, featureVector);
        if (!success)
        {
            std::runtime_error("Model inference failed");
        }
        success = postprocess(featureVector, detections);
        if (!success)
        {
            std::runtime_error("Model postprocessing failed");
        }
        return success;
    }

    bool ModelProcessor::preprocess(const cv::Mat &srcImg, cv::Mat &dstImg)
    {
        // Single batch SISO preprocessing (SBSISO)
        const auto &inputDims = m_trtEngine->getInputDims();
        assert(inputDims.size() == 1);

        cv::Size size(inputDims[0].d[1], inputDims[0].d[2]);
        return preprocess(srcImg, dstImg, size);
    }

    bool ModelProcessor::preprocess(std::vector<cv::Mat> &inputBatch, cv::Size size)
    {
        bool success = true;
        for (cv::Mat &img : inputBatch)
        {
            success = success && preprocess(img, img, size);
        }
        return success;
    }

    bool ModelProcessor::preprocess(std::vector<cv::Mat> &inputBatch)
    {
        // Multi batch SISO preprocessing (MBSISO)
        const auto &inputDims = m_trtEngine->getInputDims();
        assert(inputDims.size() == 1);

        cv::Size size(inputDims[0].d[1], inputDims[0].d[2]);
        return preprocess(inputBatch, size);
    }

    bool ModelProcessor::preprocess(std::vector<std::vector<cv::Mat>> &inputs)
    {
        // MIMO preprocessing
        bool success = true;
        const auto &inputDims = m_trtEngine->getInputDims();
        for (size_t i = 0; i < inputDims.size(); ++i)
        {
            cv::Size size(inputDims[i].d[1], inputDims[i].d[2]);
            success = success && preprocess(inputs[i], size);
        }
        return success;
    }
} // namespace trt