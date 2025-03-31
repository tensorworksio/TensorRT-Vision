#pragma once

#include "processor.hpp"
#include <opencv2/opencv.hpp>
#include <utils/vector_utils.hpp>

namespace trt
{
    template <typename OutputType, typename EngineOutput>
    ModelProcessor<OutputType, EngineOutput>::ModelProcessor(const EngineConfig &config)
    {
        // Engine options
        EngineOptions options;
        setEngineOptions(options, config.batchSize, config.precision);

        // Load engine
        engine = std::make_unique<Engine>(options);
        loadEngine(*engine, config.modelPath);
    }

    template <typename OutputType, typename EngineOutput>
    OutputType ModelProcessor<OutputType, EngineOutput>::process(const cv::Mat &image)
    {
        if (image.empty())
        {
            throw std::invalid_argument("Input image is empty");
        }

        cv::Mat processedImage;
        EngineOutput featureVector;

        if (!preprocess(image, processedImage))
        {
            throw std::runtime_error("Model preprocessing failed");
        }
        if (!engine->runInference(processedImage, featureVector))
        {
            throw std::runtime_error("Model inference failed");
        }
        return postprocess(featureVector);
    }

    template <typename OutputType, typename EngineOutput>
    std::vector<OutputType> ModelProcessor<OutputType, EngineOutput>::process(const std::vector<cv::Mat> &imageBatch)
    {
        if (imageBatch.empty())
        {
            return {};
        }

        // Pre-allocate memory
        std::vector<cv::Mat> processedBatch;
        processedBatch.reserve(imageBatch.size());

        std::vector<EngineOutput> featureBatch;
        featureBatch.reserve(imageBatch.size());

        if (!preprocess(imageBatch, processedBatch))
        {
            throw std::runtime_error("Batched model preprocessing failed");
        }

        const size_t maxBatchSize = static_cast<size_t>(engine->getOptions().maxBatchSize);

        // Process in batches
        std::vector<cv::Mat> images;
        images.reserve(maxBatchSize);

        std::vector<EngineOutput> features;
        features.reserve(maxBatchSize);

        for (size_t i = 0; i < processedBatch.size(); i += maxBatchSize)
        {
            const size_t batchSize = std::min(maxBatchSize, processedBatch.size() - i);
            images = vector_ops::slice(processedBatch, i, i + batchSize);
            if (!engine->runInference(images, features))
            {
                throw std::runtime_error("Batched model inference failed");
            }
            featureBatch.insert(featureBatch.end(),
                                std::make_move_iterator(features.begin()),
                                std::make_move_iterator(features.end()));
        }

        return postprocess(featureBatch);
    }

    template <typename OutputType, typename EngineOutput>
    bool ModelProcessor<OutputType, EngineOutput>::preprocess(const std::vector<cv::Mat> &inputBatch, std::vector<cv::Mat> &outputBatch)
    {
        outputBatch.reserve(inputBatch.size());
        cv::Mat processedImage;
        for (const auto &image : inputBatch)
        {
            if (!preprocess(image, processedImage))
            {
                return false;
            }
            outputBatch.push_back(processedImage);
        }
        return true;
    }

    template <typename OutputType, typename EngineOutput>
    std::vector<OutputType> ModelProcessor<OutputType, EngineOutput>::postprocess(const std::vector<EngineOutput> &featureBatch)
    {
        // Multi batch SISO postprocessing (MBSISO)
        std::vector<OutputType> outputs;
        outputs.reserve(featureBatch.size());

        std::transform(featureBatch.begin(), featureBatch.end(),
                       std::back_inserter(outputs),
                       [this](const auto &featureVector)
                       {
                           return postprocess(featureVector);
                       });
        return outputs;
    }

} // namespace trt
