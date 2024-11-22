#include <engine/processor.hpp>
#include <utils/vector_utils.hpp>

namespace trt
{

    ModelProcessor::ModelProcessor(const EngineConfig &config)
    {
        // Engine options
        EngineOptions options;
        setEngineOptions(options, config.batchSize, config.precision);

        // Load engine
        engine = std::make_unique<Engine>(options);
        loadEngine(*engine, config.modelPath);
    }

    bool ModelProcessor::process(const cv::Mat &image, std::vector<Detection> &detections)
    {
        // Single batch SISO inference (SBSISO)
        bool success = true;
        cv::Mat processedImage;
        std::vector<float> featureBatch;

        success = preprocess(image, processedImage);
        if (!success)
        {
            throw std::runtime_error("Model preprocessing failed.");
        }
        success = engine->runInference(processedImage, featureBatch);
        if (!success)
        {
            throw std::runtime_error("Model inference failed.");
        }
        success = postprocess(featureBatch, detections);
        if (!success)
        {
            throw std::runtime_error("Model postprocessing failed.");
        }
        return success;
    }

    bool ModelProcessor::process(const std::vector<cv::Mat> &images, std::vector<std::vector<Detection>> &detections)
    {
        // Multi batch SISO processing (MBSISO)
        bool success = true;
        std::vector<cv::Mat> processedImages;
        std::vector<std::vector<float>> features;

        success = preprocess(images, processedImages);
        if (!success)
        {
            throw std::runtime_error("Model preprocessing failed.");
        }

        int index = 0;
        size_t batchSize;
        std::vector<cv::Mat> imageBatch;
        std::vector<std::vector<float>> featureBatch;

        size_t maxBatchSize = static_cast<size_t>(engine->getOptions().maxBatchSize);
        size_t nBatch = (images.size() + maxBatchSize - 1) / maxBatchSize;

        for (size_t i = 0; i < nBatch; ++i)
        {
            batchSize = std::min(maxBatchSize, images.size() - index);
            imageBatch = vector_ops::slice(processedImages, index, index + batchSize);
            success = engine->runInference(imageBatch, featureBatch);
            if (!success)
            {
                throw std::runtime_error("Model inference failed.");
            }
            features.insert(features.end(), featureBatch.begin(), featureBatch.end());
            featureBatch.clear();
        }
        std::cout << "feature size: " << features.size() << std::endl;

        success = postprocess(features, detections);
        std::cout << "detections size: " << detections.size() << std::endl;
        if (!success)
        {
            throw std::runtime_error("Model postprocessing failed.");
        }
        return success;
    }

    bool ModelProcessor::preprocess(const cv::Mat &srcImg, cv::Mat &dstImg)
    {
        // Single batch SISO preprocessing (SBSISO)
        const auto &inputDims = engine->getInputDims();
        assert(inputDims.size() == 1);

        cv::Size size(inputDims[0].d[2], inputDims[0].d[1]);
        return preprocess(srcImg, dstImg, size);
    }

    bool ModelProcessor::preprocess(const std::vector<cv::Mat> &inputBatch, std::vector<cv::Mat> &outputBatch)
    {
        // Multi batch SISO preprocessing (MBSISO)
        const auto &inputDims = engine->getInputDims();
        assert(inputDims.size() == 1);

        cv::Size size(inputDims[0].d[2], inputDims[0].d[1]);
        return preprocess(inputBatch, outputBatch, size);
    }

    bool ModelProcessor::preprocess(const std::vector<cv::Mat> &inputBatch, std::vector<cv::Mat> &outputBatch, cv::Size size)
    {
        bool success = true;
        cv::Mat processedImage;
        for (const auto &image : inputBatch)
        {
            success = success && preprocess(image, processedImage, size);
            if (!success)
            {
                return false;
            }
            outputBatch.push_back(processedImage);
        }
        return success;
    }

    bool ModelProcessor::postprocess(std::vector<std::vector<float>> &featureBatch, std::vector<std::vector<Detection>> &detectionBatch)
    {
        // Multi batch SISO postprocessing (MBSISO)
        for (auto &features : featureBatch)
        {
            std::vector<Detection> detections;
            bool success = postprocess(features, detections);
            if (!success)
            {
                return false;
            }
            detectionBatch.push_back(detections);
        }
        return true;
    }

} // namespace trt