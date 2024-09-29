#include "engine/classifier.hpp"

namespace trt
{

    const std::string Classifier::getClassName(int class_id) const
    {
        auto config = getClassifierConfig();
        if (config.classNames.empty())
        {
            return std::to_string(class_id);
        }
        return config.classNames[class_id];
    }

    Detection Classifier::process(const cv::Mat &image)
    {
        std::vector<Detection> detections;
        ModelProcessor::process(image, detections);
        if (detections.empty())
        {
            throw std::runtime_error("Classification result should never be empty");
        }

        Detection det{};
        auto config = getClassifierConfig();
        auto maxElement = std::max_element(detections.begin(), detections.end(), [](const Detection &a, const Detection &b)
                                           { return a.probability < b.probability; });
        if (maxElement->probability >= config.confidenceThreshold)
        {
            det = *maxElement;
        }

        return det;
    }

    bool Classifier::preprocess(const cv::Mat &srcImg, cv::Mat &dstImg, cv::Size size)
    {
        // The model expects RGB image
        cv::cvtColor(srcImg, dstImg, cv::COLOR_BGR2RGB);

        // Resize and pad the image to the expected model size
        dstImg = letterbox(dstImg, size, cv::Scalar(114, 114, 114), false, true, false, 32);

        // Convert image to float
        dstImg.convertTo(dstImg, CV_32FC3, 1.f / 255.f);

        return !dstImg.empty();
    }

    bool Classifier::postprocess(std::vector<float> &featureVector, std::vector<Detection> &detections)
    {
        Detection det;
        std::vector<float> probabilities;

        auto config = getClassifierConfig();
        if (config.activation == "softmax")
        {
            probabilities = softmax(featureVector);
        }
        else
        {
            probabilities = featureVector;
        }

        for (size_t i = 0; i < probabilities.size(); ++i)
        {
            det.class_id = static_cast<int>(i);
            det.probability = probabilities[i];
            det.class_name = getClassName(det.class_id);
            detections.push_back(det);
        }

        return !detections.empty();
    }

} // namespace trt